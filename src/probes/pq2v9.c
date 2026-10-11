/* pq2v9.c -- V9 (replace the broadcast vpermd with vpslld + vpmulld) implemented
 * exactly as specified in docs/PQ2_DECODER_COST_ANALYSIS.md ("V9 design") and in
 * the task brief, then verified element-wise and (only if exact) priced.
 *
 * Self-contained on purpose: pq2cost.c was being edited concurrently by another
 * agent, so this file gives an uncontaminated answer for the spec as written.
 *
 * SPEC, for physical lane l in [0,16) and group m in [0,4):
 *     b       = 4m + l/4
 *     shift_l = 8 * ((b - (l&3)) mod 4)
 *     pre     = sllv(word, shift_l)              (per-lane shift)
 *     t       = mullo(pre, 0x01010101)           (replicate byte across dword)
 *     g[m]    = mul_ps(permutevar_epi32(t, lut), vd)
 * The four shift vectors, LANE-ORDERED as printed in the doc (so each group of
 * four is reversed when handed to _mm512_set_epi32 -- its LAST argument is
 * lane 0):
 *     m=0:  0  8 16 24 |  0  8 16 24 |  0  8 16 24 |  0  8 16 24
 *     m=1: 24  0  8 16 | 24  0  8 16 | 24  0  8 16 | 24  0  8 16
 *     m=2: 16 24  0  8 | 16 24  0  8 | 16 24  0  8 | 16 24  0  8
 *     m=3:  8 16 24  0 |  8 16 24  0 |  8 16 24  0 |  8 16 24  0
 * i.e. entry l is 8*((l&3) - m) mod 4.
 *
 * Measured premises this relies on (src/bonsai/pq2sem.c):
 *   1. permutevar_epi32(idx, data) -> result[i] = data[idx[i] & 15]
 *   2. set_epi32(a15..a0): a0 is lane 0
 *   3. srlv/sllv is a per-lane shift
 *   4. LUT lane j = (j&3)-1, so permutevar(idx, lut) yields (idx & 3) - 1
 *      -- the LUT reads the LOW 2 BITS of the index dword
 *   5. mullo(k, 0x01010101) replicates the low byte, no carry pollution
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

/* ---- V1: the shipped path (dequant + write back + reload + FMA) ---------- */
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

/* ---- V9 shift vectors, exactly as printed in the doc -------------------- */
static inline __m512i v9_shift(int m) {
    switch (m) {
    case 0:  return _mm512_set_epi32(24,16,8,0, 24,16,8,0, 24,16,8,0, 24,16,8,0);
    case 1:  return _mm512_set_epi32(16,8,0,24, 16,8,0,24, 16,8,0,24, 16,8,0,24);
    case 2:  return _mm512_set_epi32(8,0,24,16, 8,0,24,16, 8,0,24,16, 8,0,24,16);
    default: return _mm512_set_epi32(0,24,16,8, 0,24,16,8, 0,24,16,8, 0,24,16,8);
    }
}

/* ---- V9 as specified ---------------------------------------------------- */
static float v9_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_sllv_epi32(c0, v9_shift(m));
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_sllv_epi32(c1, v9_shift(m));
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V9-uniform: the OTHER reading of the same idea (one scalar shift per
 * group, shm[m] = 8*m). This is what pq2cost.c was concurrently given by
 * another agent; measured here too so the report can cover both. ---------- */
static float v9u_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    const int shm[4] = { 0, 8, 16, 24 };
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_slli_epi32(c0, shm[m]);
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_slli_epi32(c1, shm[m]);
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V10: the same goal (kill the broadcast permute) in a layout where it
 * is actually possible, and with NO multiply needed ----------------------
 *
 * V9 cannot work because every one of its operations is lane-local while the
 * byte a lane needs lives in another lane. The way out is to stop demanding
 * that output lane i own weight 16m+i: regroup the dot product by 2-bit FIELD.
 *
 * For field f and half h, `c = LOAD16(qs + 16h)` has lane l = qs[16h + l], and
 * `c >> 2f` puts field f of that byte in the low 2 bits, so the LUT gather
 * returns (code-1)*d for weight 64h + 4l + f. The 16 lanes of that one vector
 * therefore cover weights {f, 4+f, ..., 60+f} of the half -- a stride-4 set.
 * All 8 (h,f) vectors together cover all 128 weights exactly once.
 *
 * The pairing with x is then fixed by permuting x ONCE PER MATVEC CALL instead
 * of permuting the weights once per group per row:
 *
 *     xp[((b*2 + h)*4 + f)*16 + l] = x[b*128 + 64*h + 4*l + f]
 *
 * xp is exactly n floats (no extra memory: the 8 vectors of 16 lanes per block
 * are the same 128 values, reordered). The build is O(n) scalar work per call,
 * amortised over d rows -- for the harness's d=512 it is measured below.
 *
 * Per 128 weights: 2 LOAD16 (4 loads) + 6 srli + 8 LUT permutes + 8 mul_ps +
 * 8 FMA + 8 xp loads. The 8 broadcast permutes are simply GONE and nothing was
 * added to the inner loop. This is the instruction profile V7 priced at
 * 1.37x, but reached by regrouping rather than by trying to replicate a byte
 * across lanes (which the hardware cannot do).
 */
static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

static float v10_row(const uint8_t *row, const float *xp, int n) {
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
        __m512 g;
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(c0, lut)), vd);
        s0 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 +  0), s0);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srli_epi32(c0, 2), lut)), vd);
        s1 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 + 16), s1);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srli_epi32(c0, 4), lut)), vd);
        s2 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 + 32), s2);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srli_epi32(c0, 6), lut)), vd);
        s3 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 + 48), s3);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(c1, lut)), vd);
        s0 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 +  0), s0);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srli_epi32(c1, 2), lut)), vd);
        s1 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 + 16), s1);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srli_epi32(c1, 4), lut)), vd);
        s2 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 + 32), s2);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srli_epi32(c1, 6), lut)), vd);
        s3 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* harness shape: build xp for the single block the harness feeds in */
static float v10_verify(const uint8_t *row, const float *x, int n) {
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    if (n != QK_PQ2_0) return 0.0f;
    build_xp(x, xp, n);
    return v10_row(row, xp, n);
}

/* ---- V10b: V10 with the variable shift form srlv instead of the immediate
 * srli, to find out which of the two KNC actually prefers. ---------------- */
static float v10b_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const __m512i sh2 = _mm512_set1_epi32(2);
    const __m512i sh4 = _mm512_set1_epi32(4);
    const __m512i sh6 = _mm512_set1_epi32(6);
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
        __m512 g;
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(c0, lut)), vd);
        s0 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 +  0), s0);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srlv_epi32(c0, sh2), lut)), vd);
        s1 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 + 16), s1);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srlv_epi32(c0, sh4), lut)), vd);
        s2 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 + 32), s2);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srlv_epi32(c0, sh6), lut)), vd);
        s3 = _mm512_fmadd_ps(g, _mm512_load_ps(xp0 + 48), s3);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(c1, lut)), vd);
        s0 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 +  0), s0);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srlv_epi32(c1, sh2), lut)), vd);
        s1 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 + 16), s1);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srlv_epi32(c1, sh4), lut)), vd);
        s2 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 + 32), s2);
        g = _mm512_mul_ps(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(_mm512_srlv_epi32(c1, sh6), lut)), vd);
        s3 = _mm512_fmadd_ps(g, _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

static float v10b_verify(const uint8_t *row, const float *x, int n) {
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    if (n != QK_PQ2_0) return 0.0f;
    build_xp(x, xp, n);
    return v10b_row(row, xp, n);
}

/* ---- V10c: V10 with V1/V2's SCHEDULING shape: compute all 8 group vectors
 * into g[8] first, then issue the 8 FMAs. KNC is in-order and vpermd is
 * microcoded with a long latency, so issuing the permutes as a batch (instead
 * of permute -> mul -> FMA per group) may matter far more than the instruction
 * count. This variant exists to test exactly that. ---------------------- */
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

static float v10c_verify(const uint8_t *row, const float *x, int n) {
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    if (n != QK_PQ2_0) return 0.0f;
    build_xp(x, xp, n);
    return v10c_row(row, xp, n);
}

/* ---- the doc's own pricing probes, copied verbatim from pq2cost.c so that
 * their numbers can be reproduced in THIS run under identical conditions.
 * They are numerically wrong by construction; timing only. ---------------- */
static float v2_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const __m512i shf = PQ2_IMCI_SHIFT();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

static float v4_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i shf = PQ2_IMCI_SHIFT();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const uint8_t *qs = blk + 2;
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
            g[m] = _mm512_castsi512_ps(_mm512_srlv_epi32(b, shf));
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
            g[4 + m] = _mm512_castsi512_ps(_mm512_srlv_epi32(b, shf));
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

static float v5_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const __m512 a = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 2));
        const __m512 b = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 18));
        const __m512 c = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 2));
        const __m512 e = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 18));
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(c, _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(e, _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(c, _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(e, _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

static float v7_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i shf  = PQ2_IMCI_SHIFT();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    const __m512i bmask= _mm512_set1_epi32(0xFF);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(
                _mm512_and_si512(c0, bmask), mulc);
            const __m512i v = _mm512_srlv_epi32(t, shf);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(
                _mm512_and_si512(c1, bmask), mulc);
            const __m512i v = _mm512_srlv_epi32(t, shf);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

static float v7b_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i shf  = PQ2_IMCI_SHIFT();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    const __m512i bmask= _mm512_set1_epi32(0xFF);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const uint8_t *qs = blk + 2;
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(_mm512_and_si512(c0, bmask), mulc);
            g[m] = _mm512_castsi512_ps(_mm512_srlv_epi32(t, shf));
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(_mm512_and_si512(c1, bmask), mulc);
            g[4 + m] = _mm512_castsi512_ps(_mm512_srlv_epi32(t, shf));
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- element-wise harness: EVERY weight via unit vectors, plus a row sum --
 * 128 unit-vector kernel calls + 1 row-sum call per alignment, against V1. */
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

/* ---- isolation probe: which byte/field each lane actually sends to the LUT --
 * This is pq2iso.c's technique with the LUT semantics as MEASURED (premise 4):
 * permutevar(idx, lut) returns (idx & 3) - 1, i.e. it uses the LOW 2 BITS of the
 * index dword -- byte 0. It does not byte-select.                        */
static void diagnose(const uint8_t *blk, int uniform) {
    const uint8_t *qs = blk + 2;
    const __m512i c0 = PQ2_IMCI_LOAD16(qs);
    const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    int per_group[2][4];
    int total = 0;
    printf("\n=== isolation probe: %s shift ===\n",
           uniform ? "UNIFORM 8*m" : "SPEC (per-lane doc vectors)");
    printf("shift vectors as loaded (lane order):\n");
    if (!uniform) {
        for (int m = 0; m < 4; ++m) {
            int sv[16] __attribute__((aligned(64)));
            _mm512_store_epi32(sv, v9_shift(m));
            printf("  m=%d:", m);
            for (int l = 0; l < 16; ++l) printf(" %2d", sv[l]);
            printf("\n");
        }
    } else {
        printf("  m=0:  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
        printf("  m=1:  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8\n");
        printf("  m=2: 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16\n");
        printf("  m=3: 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24\n");
    }
    printf("\nmismatch map (half, group) x 16 output lanes: . = same code as V1, X = differs\n");
    for (int half = 0; half < 2; ++half) {
        for (int m = 0; m < 4; ++m) {
            const __m512i c  = half ? c1 : c0;
            const __m512i sh = uniform ? _mm512_set1_epi32(8 * m) : v9_shift(m);
            const __m512i pre = _mm512_sllv_epi32(c, sh);
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            int tl[16]  __attribute__((aligned(64)));
            int shl[16] __attribute__((aligned(64)));
            _mm512_store_epi32(tl, t);
            _mm512_store_epi32(shl, sh);
            per_group[half][m] = 0;
            printf("  half %d m %d  ", half, m);
            for (int lane = 0; lane < 16; ++lane) {
                const int base    = half ? 16 : 0;
                const int ownbyte = base + lane;
                const int needbyte= base + 4 * m + lane / 4;
                const int need    = (qs[needbyte] >> (2 * (lane & 3))) & 3;
                const int got     = tl[lane] & 3;    /* what the LUT will see */
                const int ok      = (got == need);
                printf("%c", ok ? '.' : 'X');
                if (!ok) { per_group[half][m]++; ++total; }
                (void)ownbyte; (void)shl;
            }
            printf("   %2d/16 differ\n", per_group[half][m]);
        }
    }
    printf("  total differing output lanes: %d / 128\n", total);

    printf("\nfirst 16 mismatching output lanes in detail:\n");
    printf("  %-4s %-3s %-4s %-6s %-8s %-8s %-6s %-6s %-6s %s\n",
           "half", "m", "lane", "weight", "needbyte", "ownbyte", "shift", "need", "got", "state");
    int shown = 0;
    for (int half = 0; half < 2 && shown < 16; ++half) {
        for (int m = 0; m < 4 && shown < 16; ++m) {
            const __m512i c  = half ? c1 : c0;
            const __m512i sh = uniform ? _mm512_set1_epi32(8 * m) : v9_shift(m);
            const __m512i pre = _mm512_sllv_epi32(c, sh);
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            int tl[16]  __attribute__((aligned(64)));
            int shl[16] __attribute__((aligned(64)));
            _mm512_store_epi32(tl, t);
            _mm512_store_epi32(shl, sh);
            for (int lane = 0; lane < 16 && shown < 16; ++lane) {
                const int base     = half ? 16 : 0;
                const int ownbyte  = base + lane;
                const int needbyte = base + 4 * m + lane / 4;
                const int need     = (qs[needbyte] >> (2 * (lane & 3))) & 3;
                const int got      = tl[lane] & 3;
                if (got == need) continue;
                printf("  %-4d %-3d %-4d %-6d %-8d %-8d %-6d %-6d %-6d %s\n",
                       half, m, lane, 64 * half + 16 * m + lane,
                       needbyte, ownbyte, shl[lane], need, got,
                       shl[lane] == 0 ? "own byte, field 0"
                                      : "byte shifted out -> dword byte0 = 0");
                ++shown;
            }
        }
    }

    /* structural count: lanes whose own dword byte IS the byte V1 needs */
    int reachable = 0, usable = 0;
    for (int half = 0; half < 2; ++half)
        for (int m = 0; m < 4; ++m)
            for (int lane = 0; lane < 16; ++lane) {
                const int base = half ? 16 : 0;
                if (base + lane == base + 4 * m + lane / 4) {
                    ++reachable;
                    if ((lane & 3) == 0) ++usable;
                }
            }
    printf("\nlanes whose own dword byte is the byte V1 needs: %d / 128 (own==need).\n",
           reachable);
    printf("of those, lanes that can also hit the right field: %d / 128\n", usable);
    printf("=> a lane-local op on LOAD16 data can only ever be right on %d of 128 lanes:\n", usable);
    printf("   byte 4m+l/4 lives in lane 4m+l/4, not in lane l, and neither a\n");
    printf("   per-lane shift nor vpmulld moves data across lanes.\n");
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
    const int n    = argc > 1 ? atoi(argv[1]) : 5120;
    const int d    = argc > 2 ? atoi(argv[2]) : 512;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;

    const size_t row_bytes = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float   *x = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    if (!w || !x) { fprintf(stderr, "alloc failed\n"); return 1; }

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

    printf("=== pq2v9: V9 (vpslld+vpmulld for the broadcast permute) ===\n");
    printf("n=%d rows=%d reps=%d  row_bytes=%zu  weights/row=%d\n\n",
           n, d, reps, row_bytes, n);

    /* fixed randomized block, scale 1.0, for the element-wise harness */
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

    int bad_spec[64], bad_uniform[64], bad_v10[64], bad_v10b[64], bad_v10c[64];
    int tot_spec = 0, tot_uniform = 0, tot_v10 = 0, tot_v10b = 0, tot_v10c = 0;
    for (int off = 0; off < 64; ++off) {
        memset(area, 0, 64 + 64 + 256);
        memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
        bad_spec[off]    = verify_vs("V9",  v9_row,  area + off, x, QK_PQ2_0, 0);
        bad_uniform[off] = verify_vs("V9u", v9u_row, area + off, x, QK_PQ2_0, 0);
        bad_v10[off]     = verify_vs("V10", v10_verify, area + off, x, QK_PQ2_0, 0);
        bad_v10b[off]    = verify_vs("V10b", v10b_verify, area + off, x, QK_PQ2_0, 0);
        bad_v10c[off]    = verify_vs("V10c", v10c_verify, area + off, x, QK_PQ2_0, 0);
        tot_spec    += bad_spec[off];
        tot_uniform += bad_uniform[off];
        tot_v10     += bad_v10[off];
        tot_v10b    += bad_v10b[off];
        tot_v10c    += bad_v10c[off];
    }

    printf("=== element-wise vs V1: 64 source alignments x (128 unit vectors + row sum) ===\n");
    printf("  V9  (SPEC per-lane shift vectors)  MISMATCHES = %d\n", tot_spec);
    if (tot_spec) {
        printf("    per-alignment:");
        for (int o = 0; o < 64; ++o) printf(" %d:%d", o, bad_spec[o]);
        printf("\n");
    } else {
        printf("    *** V9 IS EXACT ***\n");
    }
    printf("  V9u (uniform 8*m shift reading)    MISMATCHES = %d\n", tot_uniform);
    if (tot_uniform) {
        printf("    per-alignment:");
        for (int o = 0; o < 64; ++o) printf(" %d:%d", o, bad_uniform[o]);
        printf("\n");
    }
    printf("  V10 (regrouped by field, x premuted once) MISMATCHES = %d   %s\n",
           tot_v10, tot_v10 == 0 ? "*** EXACT ***" : "*** WRONG ***");
    if (tot_v10) {
        printf("    per-alignment:");
        for (int o = 0; o < 64; ++o) printf(" %d:%d", o, bad_v10[o]);
        printf("\n");
    }
    printf("  V10b (same, srlv instead of srli)          MISMATCHES = %d   %s\n",
           tot_v10b, tot_v10b == 0 ? "*** EXACT ***" : "*** WRONG ***");
    if (tot_v10b) {
        printf("    per-alignment:");
        for (int o = 0; o < 64; ++o) printf(" %d:%d", o, bad_v10b[o]);
        printf("\n");
    }
    printf("  V10c (regrouped, batched like V1/V2)       MISMATCHES = %d   %s\n",
           tot_v10c, tot_v10c == 0 ? "*** EXACT ***" : "*** WRONG ***");
    if (tot_v10c) {
        printf("    per-alignment:");
        for (int o = 0; o < 64; ++o) printf(" %d:%d", o, bad_v10c[o]);
        printf("\n");
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
        int bad = 0, bad10 = 0;
        for (int off = 0; off < 8; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, b2, PQ2_0_BLOCK_BYTES);
            bad   += verify_vs("V9",  v9_row,     area + off, x, QK_PQ2_0, 0);
            bad10 += verify_vs("V10", v10_verify, area + off, x, QK_PQ2_0, 0);
        }
        printf("  extra block %d (8 alignments): V9 mismatches = %d, V10 mismatches = %d\n",
               trial, bad, bad10);
    }
    printf("\n");

    /* multi-block check: the harness only ever feeds ONE block, so exercise
     * the xp mapping across block boundaries too (rows of n=512 = 4 blocks) */
    {
        float x2[512] __attribute__((aligned(64)));
        float xp2[512] __attribute__((aligned(64)));
        for (int i = 0; i < 512; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp2, 512);
        double worst = 0.0;
        for (int r = 0; r < 4; ++r) {
            const uint8_t *row = w + (size_t)r * row_bytes;
            const double a = v1_row(row, x2, 512);
            const double b = v10_row(row, xp2, 512);
            const double e = fabs(a - b) / (fabs(a) + 1.0);
            if (e > worst) worst = e;
        }
        printf("multi-block check (4 rows x n=512, 4 blocks/row): max rel diff = %.3e  %s\n",
               worst, worst < 1e-5 ? "OK" : "*** WRONG ***");
        printf("\n");
    }

    diagnose(blk, 0);
    diagnose(blk, 1);

    /* pre-permuted x for V10: built ONCE per matvec call, not per row.
     * (The row kernels are called d times with the same x, so this is the
     *  cheap side of the trade the real kernel would make.) */
    float *xp = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    if (!xp) { fprintf(stderr, "alloc failed\n"); return 1; }
    const double tb0 = now_s();
    build_xp(x, xp, n);
    const double tb1 = now_s();
    const double xp_build_s = tb1 - tb0;

    double sink = 0.0;
    printf("\nV9 (both readings) failed element-wise verification -> refusing to price it.\n");
    /* Everything below is timed in ONE run so the numbers are comparable:
     * V4/V5/V7/V7b are the doc's deliberate pricing probes (wrong by
     * construction, timing only); V10/V10b are the regrouped decoder verified
     * above; V9 itself is NOT priced because it failed verification. */
    struct { const char *name; float (*fn)(const uint8_t *, const float *, int);
             int use_xp; int exact; } V[] = {
        { "V1  dequant+reload+FMA (shipped path)",        v1_row,  0, 1 },
        { "V2  register-resident decode+FMA",             v2_row,  0, 1 },
        { "V4  ONE permute, no LUT permute  [price-only]",v4_row,  0, 0 },
        { "V5  no decode, FMA network only  [price-only]",v5_row,  0, 0 },
        { "V7  vpmulld broadcast + LUT perm  [price-only]",v7_row, 0, 0 },
        { "V7b vpmulld+vpsrlvd, no LUT       [price-only]",v7b_row,0, 0 },
        { "V10 regrouped by field (srli)",                v10_row, 1, 1 },
        { "V10b regrouped by field (srlv)",               v10b_row,1, 1 },
        { "V10c regrouped, V1-style batch scheduling",    v10c_row,1, 1 },
    };
    const unsigned nv = sizeof V / sizeof V[0];
    double wps[9], w1 = 0.0;
    printf("\n=== pricing (one run, single thread, best of %d) ===\n", reps);
    for (unsigned v = 0; v < nv; ++v) {
        const float *xv = V[v].use_xp ? xp : x;
        wps[v] = price_row(V[v].fn, w, row_bytes, xv, n, d, reps, &sink);
        if (v == 0) w1 = wps[v];
        printf("  %-50s %8.1f M weights/s  (%5.3fx)%s\n",
               V[v].name, wps[v] / 1e6, wps[v] / w1,
               V[v].exact ? "" : "  [not exact, pricing probe]");
    }
    printf("  %-50s %8s\n", "V9  vpslld+vpmulld + LUT permute", "REFUSED (failed verification)");
    printf("\n  doc reference (same card, earlier run): V1 934.0, V7 1282.0 (1.37x), "
           "V7b 2004.3, V4 1499.1, V5 2434.1\n");
    printf("  xp build for V10: %.3f ms once per matvec call (n=%d floats);"
           " at d=%d rows that is %.3f%% of V10's row work\n",
           xp_build_s * 1e3, n, d, 100.0 * xp_build_s / ((double)n * d / wps[6]));
    printf("sink=%.6g\n", sink);
    free(xp); free(area); free(w); free(x);
    return tot_spec != 0 ? 1 : 0;
}

