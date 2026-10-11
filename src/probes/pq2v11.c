/* pq2v11.c -- the real win is not the decode, it is the per-block SCALAR bath.
 *
 * Measured facts that drive this file (all from the card, this session):
 *
 *  1. Instruction throughput on this core is flat: EVERY 512-bit instruction
 *     costs ~1.645 ns (8 independent chains, pq2isa2.c).  vpermd costs exactly
 *     the same as vpaddd.  So the only currency is INSTRUCTION COUNT.
 *
 *  2. The measured decoder time divided by that flat rate gives an instruction
 *     count that matches the disassembly almost exactly:
 *        v5_row   ~30 instructions/block -> 30 * 1.77 = 53 ns  (measured 53 ns)
 *        v10c_row ~76 instructions/block -> 76 * 1.6  = 121 ns (measured 121 ns)
 *
 *  3. `icc -mmic -O3 -S` on pq2v9.c shows what those 76 instructions are:
 *        vector:  2 vmovaps + 4 vloadunpack + 6 vpsrld + 8 vpermd + 8 vmulps
 *                 + 4 vfmadd132ps + 4 vmovaps + 4 vfmadd231ps        = 40
 *        scalar:  9 (j/128 signed-division + *34 address)  + 6 (2 movb + masks
 *                 + shl + or to build the fp16) + 8 (fp16 fields, with a BRANCH)
 *                 + 2 + 5 (the normal-case tail) + 1 (store d to the stack)
 *                 + 1 + 2 (loop)                                     = 36
 *     => HALF the time of the best verified decoder is scalar bit-fiddling for
 *     the 2-byte fp16 scale, done once per 128 weights.
 *
 * Fixes implemented here:
 *   (a) pointer-increment addressing          (-13 instructions)
 *   (b) one 16-bit load for the scale         (-5)
 *   (c) the scale is converted ONCE PER ROW into a float array, and the inner
 *       loop consumes it as an embedded broadcast operand of the final FMA:
 *          vfmadd231ps (%dp){1to16}, %zmm_t, %zmm_s        (marginal cost 0)
 *   (d) `factor d out`: sum the 8 field vectors first, scale the block partial
 *       sum once, then accumulate.  8 vmulps + 8 FMA  ->  1 mulps + 7 FMA + 1 FMA
 *       (-7 instructions, and no accumulator copies so -4 vmovaps too)
 *
 * Expected: 4 + 6 + 8 + 9 = 27 vector + ~5 scalar = 32 instructions/block.
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

/* ===================== V1 reference (unchanged) ========================= */
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

/* ===================== V5 floor probe (unchanged) ======================= */
static float v5_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const __m512 a = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 2));
        const __m512 b = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 18));
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ===================== V10c (the current best) ========================== */
static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

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
        g[0] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lut)), vd);
        g[1] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0, 2), lut)), vd);
        g[2] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0, 4), lut)), vd);
        g[3] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0, 6), lut)), vd);
        g[4] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lut)), vd);
        g[5] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1, 2), lut)), vd);
        g[6] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1, 4), lut)), vd);
        g[7] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1, 6), lut)), vd);
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

/* ===================== fp16 -> fp32 alternatives ======================== */
/* branchless fast path; subnormal/zero/inf/nan go to the (correct) slow path */
static inline float fp16_fast(uint16_t h) {
    if (__builtin_expect(((h >> 10) & 0x1F) == 0 || ((h >> 10) & 0x1F) == 0x1F, 0))
        return bonsai_fp16_to_fp32(h);
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)(h & 0x8000u) << 16)
        | ((((uint32_t)(h >> 10) & 0x1Fu) + 112u) << 23)
        | ((uint32_t)(h & 0x3FFu) << 13);
    return v.f;
}

/* gather 16 half-precision scales at stride 34 and convert with the IMCI
 * FLOAT16 upconvert (vgatherdps).  offsets are element (2-byte) indices. */
static void build_drow_gather(const uint8_t *row, int nblk, float *drow) {
    const __m512i off = _mm512_set_epi32(
        15*17, 14*17, 13*17, 12*17, 11*17, 10*17, 9*17, 8*17,
         7*17,  6*17,  5*17,  4*17,  3*17,  2*17, 1*17,    0);
    int b = 0;
    for (; b + 16 <= nblk; b += 16) {
        const __m512 d = _mm512_i32extgather_ps(off, row + (size_t)34 * b,
                              _MM_UPCONV_PS_FLOAT16, 2, _MM_HINT_NONE);
        _mm512_store_ps(drow + b, d);
    }
    for (; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = bonsai_fp16_to_fp32(h);
    }
}
static void build_drow_scalar(const uint8_t *row, int nblk, float *drow) {
    for (int b = 0; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = fp16_fast(h);
    }
}

/* ===================== the lean decoders ================================ */
/* A0: 8 fields -> one partial sum t, one scale FMA into the row accumulator */
static float a0_row(const uint8_t *row, const float *xp, int n,
                    const float *drow) {
    __m512 s = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        __m512 p0,p1,p2,p3,p4,p5,p6,p7;
        p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lut));
        p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,2), lut));
        p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,4), lut));
        p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,6), lut));
        p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lut));
        p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,2), lut));
        p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,4), lut));
        p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,6), lut));
        __m512 t = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        t = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 +  16), t);
        t = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 +  32), t);
        t = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 +  48), t);
        t = _mm512_fmadd_ps(p4, _mm512_load_ps(x0 +  64), t);
        t = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 +  80), t);
        t = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 +  96), t);
        t = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 + 112), t);
        s = _mm512_fmadd_ps(t, _mm512_set1_ps(*dp), s);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
        ++dp;
    } while (--b);
    return _mm512_reduce_add_ps(s);
}

/* A1: two partial sums (4 fields each) + 1 add, then the scale FMA.
 * Same instruction count +1 but a much shorter dependency chain. */
static float a1_row(const uint8_t *row, const float *xp, int n,
                    const float *drow) {
    __m512 s = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        __m512 p0,p1,p2,p3,p4,p5,p6,p7;
        p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lut));
        p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,2), lut));
        p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,4), lut));
        p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,6), lut));
        p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lut));
        p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,2), lut));
        p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,4), lut));
        p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,6), lut));
        __m512 ta = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        __m512 tb = _mm512_mul_ps(p4, _mm512_load_ps(x0 + 64));
        ta = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 +  16), ta);
        tb = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 +  80), tb);
        ta = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 +  32), ta);
        tb = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 +  96), tb);
        ta = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 +  48), ta);
        tb = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 + 112), tb);
        s = _mm512_fmadd_ps(_mm512_add_ps(ta, tb), _mm512_set1_ps(*dp), s);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
        ++dp;
    } while (--b);
    return _mm512_reduce_add_ps(s);
}

/* A2: no scale array at all: branchless scalar conversion into a stack slot,
 * consumed as an embedded broadcast.  Tests whether the per-row pass is worth
 * more than 8 extra scalar instructions inside the loop. */
static float a2_row(const uint8_t *row, const float *xp, int n) {
    __m512 s = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const uint8_t *blk = row;
    const float *x0 = xp;
    int b = n >> 7;
    do {
        uint16_t h; memcpy(&h, blk, 2);
        const float d = fp16_fast(h);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        __m512 p0,p1,p2,p3,p4,p5,p6,p7;
        p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lut));
        p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,2), lut));
        p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,4), lut));
        p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,6), lut));
        p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lut));
        p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,2), lut));
        p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,4), lut));
        p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,6), lut));
        __m512 t = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        t = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 +  16), t);
        t = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 +  32), t);
        t = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 +  48), t);
        t = _mm512_fmadd_ps(p4, _mm512_load_ps(x0 +  64), t);
        t = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 +  80), t);
        t = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 +  96), t);
        t = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 + 112), t);
        s = _mm512_fmadd_ps(t, _mm512_set1_ps(d), s);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
    } while (--b);
    return _mm512_reduce_add_ps(s);
}

/* A3: like A0 but the scale multiply is kept per field with a 4-way
 * accumulator rotation (tests the "vmovaps copies" cost of the compiler). */
static float a3_row(const uint8_t *row, const float *xp, int n,
                    const float *drow) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512 vd = _mm512_set1_ps(*dp);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        __m512 p0,p1,p2,p3,p4,p5,p6,p7;
        p0 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lut)), vd);
        p1 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,2), lut)), vd);
        p2 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,4), lut)), vd);
        p3 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,6), lut)), vd);
        p4 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lut)), vd);
        p5 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,2), lut)), vd);
        p6 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,4), lut)), vd);
        p7 = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,6), lut)), vd);
        s0 = _mm512_fmadd_ps(p0, _mm512_load_ps(x0),      s0);
        s1 = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 + 16), s1);
        s2 = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 + 32), s2);
        s3 = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 + 48), s3);
        s0 = _mm512_fmadd_ps(p4, _mm512_load_ps(x0 + 64), s0);
        s1 = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 + 80), s1);
        s2 = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 + 96), s2);
        s3 = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 +112), s3);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
        ++dp;
    } while (--b);
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ===================== harness ========================================== */
struct kern { const char *name; int kind; };
/* kind: 0 = xp + drow, 1 = xp only, 2 = x only (V1/V5) */

static float call0(int which, const uint8_t *row, const float *x,
                   const float *xp, const float *drow, int n) {
    switch (which) {
    case 0: return v1_row(row, x, n);
    case 1: return v5_row(row, x, n);
    case 2: return v10c_row(row, xp, n);
    case 3: return a0_row(row, xp, n, drow);
    case 4: return a1_row(row, xp, n, drow);
    case 5: return a2_row(row, xp, n);
    case 6: return a3_row(row, xp, n, drow);
    default: return 0.0f;
    }
}

static int verify_one(int which, const uint8_t *block_area, const float *x,
                      int n, int verbose) {
    /* element-wise: 128 unit vectors + row sum, vs v1_row on the same block */
    int bad = 0;
    float xv[QK_PQ2_0] __attribute__((aligned(64)));
    float xp[QK_PQ2_0] __attribute__((aligned(64)));
    float drow[1] __attribute__((aligned(64)));
    uint16_t h; memcpy(&h, block_area, 2);
    drow[0] = bonsai_fp16_to_fp32(h);
    for (int jj = 0; jj < QK_PQ2_0; ++jj) {
        for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
        xv[jj] = 1.0f;
        build_xp(xv, xp, QK_PQ2_0);
        const float a = v1_row(block_area, xv, QK_PQ2_0);
        const float b = call0(which, block_area, xv, xp, drow, QK_PQ2_0);
        if (fabsf(a - b) > 1e-6f) {
            if (verbose && bad < 6)
                printf("      unit-vector mismatch at weight %d: V1=%.7g cand=%.7g\n",
                       jj, a, b);
            ++bad;
        }
    }
    build_xp(x, xp, n);
    {
        const float a = v1_row(block_area, x, n);
        const float b = call0(which, block_area, x, xp, drow, n);
        if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) {
            if (verbose) printf("      row-sum mismatch: V1=%.7g cand=%.7g\n", a, b);
            ++bad;
        }
    }
    return bad;
}

static uint32_t rng_state;
static uint32_t rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int n    = argc > 1 ? atoi(argv[1]) : 5120;
    const int d    = argc > 2 ? atoi(argv[2]) : 512;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;

    const int nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    uint8_t *w   = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float   *x   = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float   *xp  = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float   *drow= (float   *)aligned_alloc(64, ((size_t)nblk * 4 + 63) & ~(size_t)63);
    if (!w || !x || !xp || !drow) { fprintf(stderr, "alloc failed\n"); return 1; }

    rng_state = 12345u;
    for (size_t i = 0; i < row_bytes * d; ++i) w[i] = (uint8_t)(rnd() & 0xFF);
    /* give every block a RANDOM NORMAL fp16 scale so the scale path is really
     * exercised (V1's own fp16 conversion is the reference for it) */
    for (size_t r = 0; r < (size_t)d; ++r)
        for (int b = 0; b < nblk; ++b) {
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            uint16_t e = (uint16_t)(1 + (rnd() % 29));      /* 1..29 -> normal */
            uint16_t m = (uint16_t)(rnd() & 0x3FF);
            uint16_t h = (uint16_t)((e << 10) | m);
            blk[0] = (uint8_t)(h & 0xFF);
            blk[1] = (uint8_t)(h >> 8);
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;
    build_xp(x, xp, n);

    printf("=== pq2v11: kill the per-block scalar bath ===\n");
    printf("n=%d rows=%d reps=%d  nblk=%d row_bytes=%zu\n\n", n, d, reps, nblk, row_bytes);

    /* ---- gate 0: is the hardware fp16 upconvert identical to the scalar one? */
    printf("--- gate: gather/upconvert fp16->fp32 vs bonsai_fp16_to_fp32 ---\n");
    {
        uint8_t *buf = (uint8_t *)aligned_alloc(64, 34 * 16 + 64);
        float    got[16] __attribute__((aligned(64)));
        int mismatch = 0, shown = 0;
        const __m512i off = _mm512_set_epi32(
            15*17,14*17,13*17,12*17,11*17,10*17,9*17,8*17,
             7*17, 6*17, 5*17, 4*17, 3*17,2*17,1*17,0);
        int tested = 0;
        for (int g = 0; g < 4096; ++g) {
            for (int i = 0; i < 16; ++i) {
                uint16_t h = (uint16_t)(g * 16 + i);
                buf[34 * i] = (uint8_t)(h & 0xFF);
                buf[34 * i + 1] = (uint8_t)(h >> 8);
            }
            _mm512_store_ps(got, _mm512_i32extgather_ps(off, buf, _MM_UPCONV_PS_FLOAT16, 2, _MM_HINT_NONE));
            for (int i = 0; i < 16; ++i) {
                uint16_t h = (uint16_t)(g * 16 + i);
                float ref = bonsai_fp16_to_fp32(h);
                ++tested;
                if (memcmp(&ref, &got[i], 4) != 0) {
                    if (shown < 12) {
                        printf("    h=%04x ref=%08x(%g) hw=%08x(%g)\n", h,
                               *(uint32_t*)&ref, ref, *(uint32_t*)&got[i], got[i]);
                        ++shown;
                    }
                    ++mismatch;
                }
            }
            /* also compare the branchless scalar path */
        }
        printf("    gather FLOAT16 upconvert: %d/%d halves differ from the scalar reference\n",
               mismatch, tested);
        int fbad = 0, fshown = 0;
        for (uint32_t h = 0; h < 65536u; ++h) {
            float a = bonsai_fp16_to_fp32((uint16_t)h);
            float b = fp16_fast((uint16_t)h);
            if (memcmp(&a, &b, 4) != 0) {
                if (fshown < 8) { printf("    fp16_fast h=%04x ref=%g got=%g\n", h, a, b); ++fshown; }
                ++fbad;
            }
        }
        printf("    branchless fp16_fast: %d/65536 halves differ\n", fbad);
        free(buf);
    }

    /* ---- verification -------------------------------------------------- */
    printf("\n--- element-wise vs V1: 64 source alignments x (128 unit vectors + row sum) ---\n");
    uint8_t blk[QK_PQ2_0 * 2 + 96];
    uint8_t *area = (uint8_t *)aligned_alloc(64, 64 + 64 + 256);
    struct { int id; const char *name; } cand[] = {
        { 2, "V10c (current best)" },
        { 3, "A0  one t, scale array" },
        { 4, "A1  two t, scale array" },
        { 5, "A2  inline branchless scale" },
        { 6, "A3  per-field vd, 4 accum" },
    };
    for (unsigned c = 0; c < sizeof cand / sizeof cand[0]; ++c) {
        int tot = 0;
        for (int trial = 0; trial < 2; ++trial) {
            rng_state = 987654321u + 7919u * (uint32_t)trial;
            for (unsigned i = 0; i < sizeof blk; ++i) blk[i] = (uint8_t)(rnd() & 0xFF);
            /* random normal scale, different per trial */
            { uint16_t e = (uint16_t)(1 + (rnd() % 29)), m = (uint16_t)(rnd() & 0x3FF);
              uint16_t h = (uint16_t)((e << 10) | m);
              blk[0] = (uint8_t)(h & 0xFF); blk[1] = (uint8_t)(h >> 8); }
            for (int off = 0; off < 64; ++off) {
                memset(area, 0, 64 + 64 + 256);
                memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
                tot += verify_one(cand[c].id, area + off, x, QK_PQ2_0, 0);
            }
        }
        printf("  %-28s MISMATCHES = %d   %s\n", cand[c].name, tot,
               tot == 0 ? "*** EXACT ***" : "*** WRONG ***");
    }
    /* multi-block check */
    printf("\n--- multi-block (4 rows x n=512) relative diff ---\n");
    {
        float x2[512] __attribute__((aligned(64)));
        float xp2[512] __attribute__((aligned(64)));
        float dr4[4] __attribute__((aligned(64)));
        for (int i = 0; i < 512; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp2, 512);
        for (int c = 0; c < 5; ++c) {
            double worst = 0.0;
            for (int r = 0; r < 4; ++r) {
                const uint8_t *row = w + (size_t)r * row_bytes;
                for (int b = 0; b < 4; ++b) {
                    uint16_t h; memcpy(&h, row + (size_t)b * 34, 2);
                    dr4[b] = bonsai_fp16_to_fp32(h);
                }
                const double a = v1_row(row, x2, 512);
                const double b = call0(cand[c].id, row, x2, xp2, dr4, 512);
                const double e = fabs(a - b) / (fabs(a) + 1.0);
                if (e > worst) worst = e;
            }
            printf("  %-28s max rel diff = %.3e  %s\n", cand[c].name, worst,
                   worst < 1e-5 ? "OK" : "*** WRONG ***");
        }
    }

    /* ---- pricing -------------------------------------------------------- */
    printf("\n--- pricing (single thread, best of %d) ---\n", reps);
    struct { int id; const char *name; int need_drow; int build; } run[] = {
        { 0, "V1  shipped (dequant+reload)",      0, 0 },
        { 1, "V5  no decode at all   [floor]",    0, 0 },
        { 2, "V10c current best (verified)",      0, 0 },
        { 6, "A3  per-field vd, 4 accum",         1, 0 },
        { 3, "A0  one t + scale array(scalar)",   1, 1 },
        { 3, "A0  one t + scale array(gather)",   1, 2 },
        { 4, "A1  two t + scale array(gather)",   1, 2 },
        { 5, "A2  inline branchless scale",       0, 0 },
    };
    const unsigned nrun = sizeof run / sizeof run[0];
    double w1 = 0.0, base = 0.0;
    float drbuf[4096];
    double build_s[3] = {0,0,0};   /* 0 none, 1 scalar, 2 gather */
    int    build_n[3] = {0,0,0};
    double sink = 0.0;
    for (unsigned v = 0; v < nrun; ++v) {
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            double tbuild = 0.0;
            const double t0 = now_s();
            for (int i = 0; i < d; ++i) {
                const uint8_t *row = w + (size_t)i * row_bytes;
                if (run[v].build == 1) { double a = now_s(); build_drow_scalar(row, nblk, drbuf); double b2 = now_s(); tbuild += b2 - a; }
                if (run[v].build == 2) { double a = now_s(); build_drow_gather(row, nblk, drbuf); double b2 = now_s(); tbuild += b2 - a; }
                const float *dd = run[v].need_drow ? drbuf : drow;
                sink += call0(run[v].id, row, x, xp, dd, n);
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
            if (run[v].build) { build_s[run[v].build] = tbuild; build_n[run[v].build] = d; }
        }
        const double wps = (double)n * d / best;
        if (v == 0) w1 = wps;
        if (run[v].id == 2) base = wps;
        printf("  %-36s %8.1f M w/s  %6.3fx vs V1  %6.3fx vs V10c%s\n",
               run[v].name, wps / 1e6, wps / w1, base ? wps / base : 0.0,
               run[v].build ? "   (+drow build)" : "");
    }
    printf("\n  per-row scale-array build: scalar %.4f ms / %d rows, gather %.4f ms / %d rows\n",
           build_s[1] * 1e3, build_n[1], build_s[2] * 1e3, build_n[2]);
    printf("  equivalence gates above; sink=%g\n", sink);
    free(w); free(x); free(xp); free(drow); free(area);
    return 0;
}
