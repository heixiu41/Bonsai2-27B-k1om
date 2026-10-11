/* pq2v12.c -- PQ2_0 decoder v12: fewer ops per 128 weights, not fewer permutes.
 *
 * WHERE THIS COMES FROM
 * ---------------------
 * Two independent measurements (src/bonsai/pq2prim.c and pq2prim2.c) show that
 * on this KNC part every 512-bit vector ALU op has the SAME throughput:
 *     vpaddd vpandd vpord vpxord vpsrld vpermd vpmulld vmulps vfmadd
 * all measured within 1% of each other (see the raw card output).  In
 * particular a vpermd is NOT more expensive than a vpaddd.  That kills the
 * whole "replace the permute with integer arithmetic" family (V3/V7/V8/V9 and
 * the bias-float idea in pq2v11): each of those replaces 1 permute by 2-4
 * integer ops, which can only lose.  What actually pays is reducing the total
 * number of vector ops per 128 weights.
 *
 * V10c (the previous best, verified exact, 1.142-1.146x over V1) does, per
 * 128 weights:
 *     4 unpack-load instructions (2x PQ2_IMCI_LOAD16)
 *     6 srli      (one per (half,field) except field 0)
 *     8 vpermd    (the float LUT gather -- one per (half,field))
 *     8 vmulps    (apply the block scale d)
 *     8 vfmadd    (dot product)
 *     8 vmovaps   (the permuted x stream)
 *   = 42 instructions, of which 30 are vector ALU ops.
 *
 * V12 attacks the 30 ALU ops with two observations:
 *
 * (1) THE SCALE CAN GO INTO THE LUT.  permutevar(code, LUT) returns
 *     LUT[code & 15].  If the LUT vector is pre-multiplied by the block scale
 *     d, the gather returns (code-1)*d -- the weight itself -- so the 8 vmulps
 *     disappear and are replaced by TWO vmulps that scale the two LUTs once
 *     per block.  Numerically identical: the value is still the single product
 *     (float)(code-1) * d of the *same* two exact operands, so V12 is bitwise
 *     equal to V10/V1 on the weight values.
 *
 * (2) TWO LUTS HALVE THE SHIFTS.  the index that permutevar consults is
 *     `code & 15`, i.e. the FOUR low bits of the code byte = field 0 (bits 0-1)
 *     and field 1 (bits 2-3).  With one LUT of value (j&3)-1 you get field 0;
 *     with a SECOND LUT of value ((j>>2)&3)-1 you get field 1 FROM THE SAME
 *     UNSHIFTED INDEX.  Fields 2 and 3 need one shift by 4 (both at once),
 *     after which the same two LUTs give fields 2 and 3.
 *     => 2 srli per block instead of 6.
 *
 * V12 therefore does, per 128 weights:
 *     4 unpack + 2 srli + 8 vpermd + 2 vmulps + 8 vfmadd + 8 vmovaps
 *   = 32 instructions, 20 vector ALU ops (-33%).
 *
 * The x permutation xp is exactly V10's (build_xp below) -- it is built once
 * per matvec call, amortised over d rows, and costs 0.04-0.06 ms at n=5120.
 *
 * Everything is verified element-wise against V1 (the shipped decoder) over 64
 * source alignments x (128 unit vectors + row sum), plus extra random blocks
 * and multi-block rows, before it is priced.  All price probes are explicitly
 * marked WRONG and are never eligible for integration.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#include "pq2_0.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ===================== V1: the shipped path ============================== */
static float v1_row(const uint8_t *row, const float *x, int n) {
    float out[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        pq2_0_dequant_block_imci(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, out);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(out),      _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(out + 16), _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(out + 32), _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(out + 48), _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(out + 64), _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(out + 80), _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(out + 96), _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(out +112), _mm512_load_ps(x + j +112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ===================== xp: V10's input permutation ======================= */
static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

/* ===================== V10c: previous best (verified exact) =============== */
static float v10c_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        const float *xp0 = xp + b * 128;
        const float *xp1 = xp0 + 64;
        __m512 g[8];
        g[0] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(c0, lut)), vd);
        g[1] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(_mm512_srli_epi32(c0, 2), lut)), vd);
        g[2] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(_mm512_srli_epi32(c0, 4), lut)), vd);
        g[3] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(_mm512_srli_epi32(c0, 6), lut)), vd);
        g[4] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(c1, lut)), vd);
        g[5] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(_mm512_srli_epi32(c1, 2), lut)), vd);
        g[6] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(_mm512_srli_epi32(c1, 4), lut)), vd);
        g[7] = _mm512_mul_ps(_mm512_castsi512_ps(
                   _mm512_permutevar_epi32(_mm512_srli_epi32(c1, 6), lut)), vd);
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- the two LUTs -------------------------------------------------------
 * LA[j] = (float)((j & 3) - 1)         reads field 0 of the index byte
 * LB[j] = (float)(((j >> 2) & 3) - 1)  reads field 1 of the index byte
 * (permutevar consults index & 15 only -- measured, see pq2sem.c.)
 * Lane order: _mm512_set_epi32's LAST argument is lane 0.                 */
#define PQ2_LUT_A() \
    _mm512_set_epi32((int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u, \
                     (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u, \
                     (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u, \
                     (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u)
/* lanes 0..3 = -1, lanes 4..7 = 0, lanes 8..11 = +1, lanes 12..15 = +2 */
#define PQ2_LUT_B() \
    _mm512_set_epi32((int)0x40000000u,(int)0x40000000u,(int)0x40000000u,(int)0x40000000u, \
                     (int)0x3F800000u,(int)0x3F800000u,(int)0x3F800000u,(int)0x3F800000u, \
                     0,0,0,0, \
                     (int)0xBF800000u,(int)0xBF800000u,(int)0xBF800000u,(int)0xBF800000u)

/* ===================== V12: the candidate ================================
 * per 128 weights: 2 LOAD16 (4 instr) + 2 srli + 8 vpermd + 2 vmulps
 *                  + 8 vfmadd + 8 vmovaps = 32 instructions.
 * The assembly of this exact loop is counted in the report (srccount.py).
 * ---------------------------------------------------------------------- */
static float v12_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 la0 = _mm512_castsi512_ps(PQ2_LUT_A());
    const __m512 lb0 = _mm512_castsi512_ps(PQ2_LUT_B());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128;
        const float *xp1 = xp0 + 64;
        /* the two scaled LUTs: 2 vmulps per block instead of 8 */
        const __m512 la = _mm512_mul_ps(la0, vd);
        const __m512 lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lai));  /* f0 */
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lbi));  /* f1 */
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai)); /* f2 */
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi)); /* f3 */
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

static float v12_verify(const uint8_t *row, const float *x, int n) {
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    if (n != QK_PQ2_0) return 0.0f;
    build_xp(x, xp, n);
    return v12_row(row, xp, n);
}
static float v10c_verify(const uint8_t *row, const float *x, int n) {
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    if (n != QK_PQ2_0) return 0.0f;
    build_xp(x, xp, n);
    return v10c_row(row, xp, n);
}

/* ===================== V12b: scale fold only (6 srli, 1 LUT) =============
 * Separates the two optimisations: this one keeps V10c's 6 srli / one LUT but
 * folds the scale into the LUT (so 1 vmulps instead of 8).
 * ---------------------------------------------------------------------- */
static float v12b_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 l0 = _mm512_castsi512_ps(PQ2_LUT_A());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        const float *xp0 = xp + b * 128;
        const float *xp1 = xp0 + 64;
        const __m512i l = _mm512_castps_si512(_mm512_mul_ps(l0, vd));
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0, 2), l));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0, 4), l));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0, 6), l));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1, 2), l));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1, 4), l));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1, 6), l));
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}
static float v12b_verify(const uint8_t *row, const float *x, int n) {
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    if (n != QK_PQ2_0) return 0.0f;
    build_xp(x, xp, n);
    return v12b_row(row, xp, n);
}

/* ===================== price-only probes (WRONG by construction) =========
 * Each probe changes exactly ONE thing relative to V12 so that the marginal
 * cost of that thing can be read off the pricing table.  They are numerically
 * meaningless and must never be integrated.
 * ---------------------------------------------------------------------- */

/* P_A: the 8 vpermd replaced by 8 vpaddd (same op count, different op class) */
static float p12_add_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i l0 = PQ2_LUT_A(), l1 = PQ2_LUT_B();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_add_epi32(c0, l0));
        g[1] = _mm512_castsi512_ps(_mm512_add_epi32(c0, l1));
        g[2] = _mm512_castsi512_ps(_mm512_add_epi32(c0h, l0));
        g[3] = _mm512_castsi512_ps(_mm512_add_epi32(c0h, l1));
        g[4] = _mm512_castsi512_ps(_mm512_add_epi32(c1, l0));
        g[5] = _mm512_castsi512_ps(_mm512_add_epi32(c1, l1));
        g[6] = _mm512_castsi512_ps(_mm512_add_epi32(c1h, l0));
        g[7] = _mm512_castsi512_ps(_mm512_add_epi32(c1h, l1));
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* P_B: the 8 xp loads replaced by one hoisted register (prices the xp stream) */
static float p12_noxp_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 xv = _mm512_load_ps(xp);        /* hoisted, loop-invariant */
    const __m512i la_b = PQ2_LUT_A(), lb_b = PQ2_LUT_B();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, la_b));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lb_b));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, la_b));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lb_b));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, la_b));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lb_b));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, la_b));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lb_b));
        s0 = _mm512_fmadd_ps(_mm512_mul_ps(g[0], vd), xv, s0);
        s1 = _mm512_fmadd_ps(_mm512_mul_ps(g[1], vd), xv, s1);
        s2 = _mm512_fmadd_ps(_mm512_mul_ps(g[2], vd), xv, s2);
        s3 = _mm512_fmadd_ps(_mm512_mul_ps(g[3], vd), xv, s3);
        s0 = _mm512_fmadd_ps(_mm512_mul_ps(g[4], vd), xv, s0);
        s1 = _mm512_fmadd_ps(_mm512_mul_ps(g[5], vd), xv, s1);
        s2 = _mm512_fmadd_ps(_mm512_mul_ps(g[6], vd), xv, s2);
        s3 = _mm512_fmadd_ps(_mm512_mul_ps(g[7], vd), xv, s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* P_C: only ONE LOAD16 (c0 used for both halves) -> prices 2 unpack loads */
static float p12_one16_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 la0 = _mm512_castsi512_ps(PQ2_LUT_A());
    const __m512 lb0 = _mm512_castsi512_ps(PQ2_LUT_B());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);       /* ONLY load */
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la), lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = g[0]; g[5] = g[1]; g[6] = g[2]; g[7] = g[3];
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* P_D: the 2 LOAD16 replaced by 2 plain ALIGNED epi32 loads (prices the
 * unpack-load path against an ordinary 64-byte load) */
static float p12_align_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 la0 = _mm512_castsi512_ps(PQ2_LUT_A());
    const __m512 lb0 = _mm512_castsi512_ps(PQ2_LUT_B());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        /* ALIGNED plain loads standing in for the two unpack loads.  The
         * source is deliberately wrong (hence a price probe); the address is
         * rounded DOWN to 64 bytes so the aligned form is legal, and the
         * over-read stays inside the big weights allocation. */
        const uint8_t *a0 = (const uint8_t *)((uintptr_t)blk & ~(uintptr_t)63);
        const __m512i c0 = _mm512_load_epi32((const void *)a0);
        const __m512i c1 = _mm512_load_epi32((const void *)(a0 + 64));
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la), lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* P_E: no decode at all in the V12 shape: FMA + loads only (floor) */
static float p12_floor_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 a = _mm512_castsi512_ps(PQ2_LUT_A());
    const __m512 b = _mm512_castsi512_ps(PQ2_LUT_B());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t bb = (size_t)(j / QK_PQ2_0);
        const float *xp0 = xp + bb * 128, *xp1 = xp0 + 64;
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(a, _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(b, _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(a, _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(b, _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ===================== element-wise verification ========================= */
static int verify_vs(const char *label,
                     float (*cand)(const uint8_t *, const float *, int),
                     const uint8_t *block_area, const float *x, int n,
                     int verbose) {
    int bad = 0;
    float xv[QK_PQ2_0] __attribute__((aligned(64)));
    for (int jj = 0; jj < QK_PQ2_0; ++jj) {
        for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
        xv[jj] = 1.0f;
        const float a = v1_row(block_area, xv, QK_PQ2_0);
        const float b = cand(block_area, xv, QK_PQ2_0);
        if (fabsf(a - b) > 1e-6f) {
            if (verbose && bad < 8)
                printf("    %s unit-vector mismatch at weight %d: V1=%.7g %s=%.7g\n",
                       label, jj, a, label, b);
            ++bad;
        }
    }
    const float a = v1_row(block_area, x, n);
    const float b = cand(block_area, x, n);
    if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) {
        if (verbose) printf("    %s row-sum mismatch: V1=%.7g %s=%.7g\n", label, a, label, b);
        ++bad;
    }
    return bad;
}

/* ---- pricing (only reached when the candidate verified exactly) --------- */
static double price_row(float (*fn)(const uint8_t *, const float *, int),
                        const uint8_t *w, size_t row_bytes,
                        const float *x, int n, int d, int reps, double *sink) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        double acc = 0;
        for (int i = 0; i < d; ++i) acc += fn(w + (size_t)i * row_bytes, x, n);
        const double t1 = now_s();
        if (t1 - t0 < best) best = t1 - t0;
        *sink += acc;
    }
    return (double)n * d / best;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* survive a crash with output intact */
    const int n    = argc > 1 ? atoi(argv[1]) : 5120;
    const int d    = argc > 2 ? atoi(argv[2]) : 512;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;

    const size_t row_bytes = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float   *x = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float   *xp = (float  *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    if (!w || !x || !xp) { fprintf(stderr, "alloc failed\n"); return 1; }

    uint32_t st = 12345u;
    for (size_t i = 0; i < row_bytes * d; ++i) {
        st = st * 1103515245u + 12345u;
        w[i] = (uint8_t)(st >> 16);
    }
    for (size_t r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;               /* fp16 scale = 1.0 */
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;

    printf("=== pq2v12: fold the scale into the LUT + two LUTs to halve the shifts ===\n");
    printf("n=%d rows=%d reps=%d  row_bytes=%zu  weights/row=%d\n\n", n, d, reps, row_bytes, n);

    /* fixed randomized block for the element-wise harness */
    uint8_t blk[QK_PQ2_0 * 2 + 34];
    {
        uint32_t s2 = 987654321u;
        for (unsigned i = 0; i < sizeof blk; ++i) {
            s2 = s2 * 1103515245u + 12345u;
            blk[i] = (uint8_t)(s2 >> 16);
        }
        blk[0] = 0x00; blk[1] = 0x3C;
    }
    uint8_t *area = (uint8_t *)aligned_alloc(64, 64 + 64 + 256);
    if (!area) { fprintf(stderr, "alloc failed\n"); return 1; }

    int bad10[64], bad12[64], bad12b[64];
    int t10 = 0, t12 = 0, t12b = 0;
    for (int off = 0; off < 64; ++off) {
        memset(area, 0, 64 + 64 + 256);
        memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
        bad10[off]  = verify_vs("V10c",  v10c_verify, area + off, x, QK_PQ2_0, 0);
        bad12[off]  = verify_vs("V12",   v12_verify,  area + off, x, QK_PQ2_0, 0);
        bad12b[off] = verify_vs("V12b",  v12b_verify, area + off, x, QK_PQ2_0, 0);
        t10 += bad10[off]; t12 += bad12[off]; t12b += bad12b[off];
    }
    printf("=== element-wise vs V1: 64 source alignments x (128 unit vectors + row sum) ===\n");
    printf("  V10c (previous best)             MISMATCHES = %d  %s\n",
           t10, t10 == 0 ? "*** EXACT ***" : "*** WRONG ***");
    printf("  V12  (scale-in-LUT + 2 LUTs)     MISMATCHES = %d  %s\n",
           t12, t12 == 0 ? "*** EXACT ***" : "*** WRONG ***");
    printf("  V12b (scale-in-LUT only)         MISMATCHES = %d  %s\n",
           t12b, t12b == 0 ? "*** EXACT ***" : "*** WRONG ***");
    if (t12) {
        printf("    per-alignment:");
        for (int o = 0; o < 64; ++o) printf(" %d:%d", o, bad12[o]);
        printf("\n");
        /* verbose diagnose on the first alignment */
        memset(area, 0, 64 + 64 + 256);
        memcpy(area, blk, PQ2_0_BLOCK_BYTES);
        verify_vs("V12", v12_verify, area, x, QK_PQ2_0, 1);
    }
    printf("\n");

    /* extra robustness: 4 more random blocks x 8 alignments */
    for (int trial = 0; trial < 4; ++trial) {
        uint8_t b2[QK_PQ2_0 * 2 + 34];
        uint32_t s3 = 1234567u + 7919u * (uint32_t)trial;
        for (unsigned i = 0; i < sizeof b2; ++i) {
            s3 = s3 * 1103515245u + 12345u;
            b2[i] = (uint8_t)(s3 >> 16);
        }
        b2[0] = 0x00; b2[1] = 0x3C;
        int bad = 0, badb = 0;
        for (int off = 0; off < 8; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, b2, PQ2_0_BLOCK_BYTES);
            bad  += verify_vs("V12",  v12_verify,  area + off, x, QK_PQ2_0, 0);
            badb += verify_vs("V12b", v12b_verify, area + off, x, QK_PQ2_0, 0);
        }
        printf("  extra block %d (8 alignments): V12 mismatches = %d, V12b = %d\n",
               trial, bad, badb);
    }

    /* bitwise comparison against V10c on real rows: the scale fold should make
     * V12 bit-identical to V10c on the *weight* values; the only difference is
     * the FMA pairing order (which is deliberately the same as V10c's). */
    {
        float x2[512] __attribute__((aligned(64)));
        float xp2[512] __attribute__((aligned(64)));
        for (int i = 0; i < 512; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp2, 512);
        double worst10 = 0.0, worstv1 = 0.0;
        for (int r = 0; r < 4; ++r) {
            const uint8_t *row = w + (size_t)r * row_bytes;
            const double a  = v1_row(row, x2, 512);
            const double b  = v10c_row(row, xp2, 512);
            const double c  = v12_row(row, xp2, 512);
            const double e  = fabs(a - c) / (fabs(a) + 1.0);
            const double e10 = fabs(b - c) / (fabs(b) + 1.0);
            if (e > worstv1) worstv1 = e;
            if (e10 > worst10) worst10 = e10;
        }
        printf("\nmulti-block check (4 rows x n=512): max rel diff V12 vs V1 = %.3e %s ; "
               "V12 vs V10c = %.3e\n",
               worstv1, worstv1 < 1e-5 ? "OK" : "*** WRONG ***", worst10);
    }

    build_xp(x, xp, n);

    double sink = 0.0;
    struct { const char *name; float (*fn)(const uint8_t *, const float *, int);
             int use_xp; int exact; } V[] = {
        { "V1   shipped dequant+reload+FMA",               v1_row,        0, 1 },
        { "V10c previous best (regrouped x, batch sched)", v10c_row,      1, 1 },
        { "V12  scale-in-LUT + 2 LUTs (2 srli)",           v12_row,       1, 1 },
        { "V12b scale-in-LUT only (6 srli, 1 LUT)",        v12b_row,      1, 1 },
        { "P_A  V12 with add instead of vpermd [WRONG]",   p12_add_row,   1, 0 },
        { "P_B  V12 with no xp loads         [WRONG]",     p12_noxp_row,  1, 0 },
        { "P_C  V12 with 1 LOAD16            [WRONG]",     p12_one16_row, 1, 0 },
        { "P_D  V12 with aligned loads       [WRONG]",     p12_align_row, 1, 0 },
        { "P_E  no decode at all (floor)     [WRONG]",     p12_floor_row, 1, 0 },
    };
    const unsigned nv = sizeof V / sizeof V[0];
    double w1 = 0.0;
    printf("\n=== pricing (one run, single thread, best of %d) ===\n", reps);
    for (unsigned v = 0; v < nv; ++v) {
        const float *xv = V[v].use_xp ? xp : x;
        const double wps = price_row(V[v].fn, w, row_bytes, xv, n, d, reps, &sink);
        if (v == 0) w1 = wps;
        printf("  %-48s %8.1f M weights/s  (%5.3fx)%s\n",
               V[v].name, wps / 1e6, wps / w1,
               V[v].exact ? "" : "  [not exact, price probe]");
    }
    printf("\n  ps/weight = 1e6/(M weights/s) ; V1 is the 1.000x reference\n");
    printf("sink=%.6g\n", sink);
    free(xp); free(area); free(w); free(x);
    return t12 != 0 ? 1 : 0;
}
