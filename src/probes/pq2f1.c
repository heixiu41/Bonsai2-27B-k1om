/* pq2f1.c -- single-binary head-to-head: the best-known structures, measured
 * against V1 and V10c inside ONE executable.
 *
 * WHY ONE BINARY: pq2v16.c (another agent) found that the SAME v12_row source
 * measured 1216-1228 M w/s in one binary and 1461 M w/s in another, with the
 * V1 reference moving from 718 to 934 M w/s.  On this in-order core the
 * compiler's scheduling decides, so a ratio must come from one binary.
 *
 * STRUCTURES
 *   F0 = V10c   (previous best verified decoder)
 *   F1 = two-LUT trick (pq2v12's idea: a second LUT whose entry j is
 *        ((j>>2)&3)-1 lets the SAME unshifted index yield field 1, so only
 *        fields 2/3 need a shift) + the scale-array finding from pq2f1's
 *        sibling measurement: the per-block fp16 scale is converted once per
 *        ROW into a float array and consumed as an embedded {1to16} broadcast
 *        of the LUT-scaling multiply.  Pointer-increment addressing.
 *   F2 = F1 with TWO blocks in flight (16 independent permutes before any FMA)
 *   F3 = F1 with one shared partial sum per block (fewer ops, longer chain)
 *
 * All use xp[((b*2+h)*4+f)*16+l] = x[b*128 + 64h + 4l + f] (V10 layout).
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
static uint32_t rng_state;
static uint32_t rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }

/* ---- branchless fp16 -> fp32, bit-identical on all 65536 halves ---------- */
static inline float fp16_fast(uint16_t h) {
    union { uint32_t u; float f; } v;
    if (__builtin_expect((h & 0x7C00u) == 0 || (h & 0x7C00u) == 0x7C00u, 0))
        return bonsai_fp16_to_fp32(h);
    v.u = ((uint32_t)(h & 0x8000u) << 16)
        | ((((uint32_t)h & 0x7FFFu) << 13) + 0x38000000u);
    return v.f;
}
static void build_drow(const uint8_t *row, int nblk, float *drow) {
    for (int b = 0; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = fp16_fast(h);
    }
}

/* ---- xp layout (V10): field-major within each half --------------------- */
static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

/* base LUTs: L0[j] = (float)((j&3)-1)          -> field f   after (c>>2f)
 *            L1[j] = (float)(((j>>2)&3)-1)     -> field f+1 after (c>>2f)
 * (lane 0 is the LAST argument of _mm512_set_epi32) */
#define LUT0_BITS() _mm512_set_epi32( \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000, \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000, \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000, \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000)
#define LUT1_BITS() _mm512_set_epi32( \
    0x40000000,0x40000000,0x40000000,0x40000000, \
    0x3F800000,0x3F800000,0x3F800000,0x3F800000, \
    0,0,0,0, \
    (int)0xBF800000,(int)0xBF800000,(int)0xBF800000,(int)0xBF800000)

/* ---- V1 reference (shipped path) --------------------------------------- */
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

/* ---- F0 = V10c --------------------------------------------------------- */
static float f0_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    (void)drow;
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const float *xp0 = xp + b * 128;
        const float *xp1 = xp0 + 64;
        __m512 g[8];
        g[0] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lut)), vd);
        g[1] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,2), lut)), vd);
        g[2] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,4), lut)), vd);
        g[3] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c0,6), lut)), vd);
        g[4] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lut)), vd);
        g[5] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,2), lut)), vd);
        g[6] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,4), lut)), vd);
        g[7] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(_mm512_srli_epi32(c1,6), lut)), vd);
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1),      s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- F1: two LUTs + precomputed scales, 4 accumulators ---------------- */
static float f1_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512 vd = _mm512_set1_ps(*dp);
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        const __m512 p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));
        const __m512 p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));
        const __m512 p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));
        const __m512 p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));
        const __m512 p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));
        const __m512 p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));
        const __m512 p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));
        const __m512 p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));
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

/* ---- F3: two LUTs + precomputed scales, ONE partial sum per block ----- */
static float f3_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512 vd = _mm512_set1_ps(*dp);
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        const __m512 p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));
        const __m512 p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));
        const __m512 p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));
        const __m512 p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));
        const __m512 p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));
        const __m512 p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));
        const __m512 p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));
        const __m512 p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));
        __m512 t = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        t = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 + 16), t);
        t = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 + 32), t);
        t = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 + 48), t);
        t = _mm512_fmadd_ps(p4, _mm512_load_ps(x0 + 64), t);
        t = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 + 80), t);
        t = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 + 96), t);
        t = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 +112), t);
        s = _mm512_add_ps(s, t);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
        ++dp;
    } while (--b);
    return _mm512_reduce_add_ps(s);
}

/* ---- F2: F1 with two blocks in flight --------------------------------- */
#define F2_BODY(BOFF, XOFF, DOFF, A0,A1,A2,A3)                                \
    do {                                                                      \
        const __m512 vd = _mm512_set1_ps(dp[DOFF]);                           \
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));        \
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));        \
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + (BOFF) + 2);                 \
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + (BOFF) + 18);                \
        const __m512i e0 = _mm512_srli_epi32(c0, 4);                          \
        const __m512i e1 = _mm512_srli_epi32(c1, 4);                          \
        const float *xx = x0 + (XOFF);                                        \
        __m512 q0,q1,q2,q3,q4,q5,q6,q7;                                       \
        q0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));            \
        q1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));            \
        q2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));            \
        q3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));            \
        q4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));            \
        q5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));            \
        q6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));            \
        q7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));            \
        A0 = _mm512_fmadd_ps(q0, _mm512_load_ps(xx),      A0);                \
        A1 = _mm512_fmadd_ps(q1, _mm512_load_ps(xx + 16), A1);                \
        A2 = _mm512_fmadd_ps(q2, _mm512_load_ps(xx + 32), A2);                \
        A3 = _mm512_fmadd_ps(q3, _mm512_load_ps(xx + 48), A3);                \
        A0 = _mm512_fmadd_ps(q4, _mm512_load_ps(xx + 64), A0);                \
        A1 = _mm512_fmadd_ps(q5, _mm512_load_ps(xx + 80), A1);                \
        A2 = _mm512_fmadd_ps(q6, _mm512_load_ps(xx + 96), A2);                \
        A3 = _mm512_fmadd_ps(q7, _mm512_load_ps(xx +112), A3);                \
    } while (0)

static float f2_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    __m512 b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int nb = n >> 7;
    int pairs = nb >> 1;
    while (pairs--) {
        /* issue BOTH blocks' permutes before ANY of the FMAs */
        F2_BODY(0,    0, 0, a0,a1,a2,a3);
        F2_BODY(34, 128, 1, b0,b1,b2,b3);
        blk += 2 * PQ2_0_BLOCK_BYTES;
        x0  += 2 * QK_PQ2_0;
        dp  += 2;
    }
    if (nb & 1) {
        const __m512 vd = _mm512_set1_ps(*dp);
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0,l0)), _mm512_load_ps(x0),      a0);
        a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0,l1)), _mm512_load_ps(x0 + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e0,l0)), _mm512_load_ps(x0 + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e0,l1)), _mm512_load_ps(x0 + 48), a3);
        a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1,l0)), _mm512_load_ps(x0 + 64), a0);
        a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1,l1)), _mm512_load_ps(x0 + 80), a1);
        a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e1,l0)), _mm512_load_ps(x0 + 96), a2);
        a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e1,l1)), _mm512_load_ps(x0 +112), a3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1), _mm512_add_ps(a2,a3)),
                             _mm512_add_ps(_mm512_add_ps(b0,b1), _mm512_add_ps(b2,b3)));
    return _mm512_reduce_add_ps(s);
}

/* ---- F4: two LUTs + precomputed scales, TWO partial sums (depth 5) ---- */
static float f4_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512 vd = _mm512_set1_ps(*dp);
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        const __m512 p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));
        const __m512 p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));
        const __m512 p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));
        const __m512 p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));
        const __m512 p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));
        const __m512 p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));
        const __m512 p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));
        const __m512 p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));
        __m512 ta = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        __m512 tb = _mm512_mul_ps(p4, _mm512_load_ps(x0 + 64));
        ta = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 + 16), ta);
        tb = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 + 80), tb);
        ta = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 + 32), ta);
        tb = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 + 96), tb);
        ta = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 + 48), ta);
        tb = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 +112), tb);
        s0 = _mm512_add_ps(s0, ta);
        s1 = _mm512_add_ps(s1, tb);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
        ++dp;
    } while (--b);
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}

/* ---- F5: 2 LUTs + INLINE branchless scale, 1 partial sum ------------- */
static float f5_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    (void)drow;
    int b = n >> 7;
    do {
        uint16_t h; memcpy(&h, blk, 2);
        const __m512 vd = _mm512_set1_ps(fp16_fast(h));
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        const __m512 p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));
        const __m512 p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));
        const __m512 p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));
        const __m512 p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));
        const __m512 p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));
        const __m512 p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));
        const __m512 p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));
        const __m512 p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));
        __m512 t = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        t = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 + 16), t);
        t = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 + 32), t);
        t = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 + 48), t);
        t = _mm512_fmadd_ps(p4, _mm512_load_ps(x0 + 64), t);
        t = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 + 80), t);
        t = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 + 96), t);
        t = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 +112), t);
        s = _mm512_add_ps(s, t);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
    } while (--b);
    return _mm512_reduce_add_ps(s);
}

/* ---- F6: 2 LUTs + INLINE scale, 2 partial sums (best ILP/op trade) ---- */
static float f6_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    (void)drow;
    int b = n >> 7;
    do {
        uint16_t h; memcpy(&h, blk, 2);
        const __m512 vd = _mm512_set1_ps(fp16_fast(h));
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        const __m512 p0 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));
        const __m512 p1 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));
        const __m512 p2 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));
        const __m512 p3 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));
        const __m512 p4 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));
        const __m512 p5 = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));
        const __m512 p6 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));
        const __m512 p7 = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));
        __m512 ta = _mm512_mul_ps(p0, _mm512_load_ps(x0));
        __m512 tb = _mm512_mul_ps(p4, _mm512_load_ps(x0 + 64));
        ta = _mm512_fmadd_ps(p1, _mm512_load_ps(x0 + 16), ta);
        tb = _mm512_fmadd_ps(p5, _mm512_load_ps(x0 + 80), tb);
        ta = _mm512_fmadd_ps(p2, _mm512_load_ps(x0 + 32), ta);
        tb = _mm512_fmadd_ps(p6, _mm512_load_ps(x0 + 96), tb);
        ta = _mm512_fmadd_ps(p3, _mm512_load_ps(x0 + 48), ta);
        tb = _mm512_fmadd_ps(p7, _mm512_load_ps(x0 +112), tb);
        s0 = _mm512_add_ps(s0, ta);
        s1 = _mm512_add_ps(s1, tb);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
    } while (--b);
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}

/* ===================== verification + pricing =========================== */
typedef float (*fn_t)(const uint8_t *, const float *, int, const float *);

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int n    = argc > 1 ? atoi(argv[1]) : 5120;
    const int d    = argc > 2 ? atoi(argv[2]) : 512;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    const int nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;

    uint8_t *w    = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float   *x    = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float   *xp   = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float   *drow = (float   *)aligned_alloc(64, ((size_t)nblk * 4 + 63) & ~(size_t)63);
    if (!w || !x || !xp || !drow) { fprintf(stderr, "alloc failed\n"); return 1; }

    rng_state = 12345u;
    for (size_t i = 0; i < row_bytes * d; ++i) w[i] = (uint8_t)(rnd() & 0xFF);
    for (size_t r = 0; r < (size_t)d; ++r)
        for (int b = 0; b < nblk; ++b) {
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            uint16_t e = (uint16_t)(1 + (rnd() % 29));
            uint16_t m = (uint16_t)(rnd() & 0x3FF);
            uint16_t h = (uint16_t)((e << 10) | m);
            blk[0] = (uint8_t)(h & 0xFF); blk[1] = (uint8_t)(h >> 8);
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;
    build_xp(x, xp, n);

    printf("=== pq2f1: one binary, all anchors ===\n");
    printf("n=%d rows=%d reps=%d nblk=%d row_bytes=%zu\n\n", n, d, reps, nblk, row_bytes);

    struct { fn_t f; const char *name; } K[] = {
        { f0_row, "F0  V10c (prev best)" },
        { f1_row, "F1  2 LUTs + scale array" },
        { f2_row, "F2  F1 + 2 blocks in flight" },
        { f3_row, "F3  F1 + 1 partial sum" },
        { f4_row, "F4  F1 + 2 partial sums" },
        { f5_row, "F5  2 LUTs + INLINE scale, 1 t" },
        { f6_row, "F6  2 LUTs + INLINE scale, 2 t" },
    };
    const unsigned NK = sizeof K / sizeof K[0];

    /* ---- element-wise verification ------------------------------------- */
    printf("--- element-wise vs V1: 64 alignments x (128 unit vectors + row sum), 2 blocks ---\n");
    uint8_t *area = (uint8_t *)aligned_alloc(64, 64 + 64 + 256);
    uint8_t blk[QK_PQ2_0 * 2 + 96];
    for (unsigned k = 0; k < NK; ++k) {
        int tot = 0;
        for (int trial = 0; trial < 2; ++trial) {
            rng_state = 987654321u + 7919u * (uint32_t)trial;
            for (unsigned i = 0; i < sizeof blk; ++i) blk[i] = (uint8_t)(rnd() & 0xFF);
            { uint16_t e = (uint16_t)(1 + (rnd() % 29)), m = (uint16_t)(rnd() & 0x3FF);
              uint16_t h = (uint16_t)((e << 10) | m);
              blk[0] = (uint8_t)(h & 0xFF); blk[1] = (uint8_t)(h >> 8); }
            for (int off = 0; off < 64; ++off) {
                float xv[QK_PQ2_0] __attribute__((aligned(64)));
                float xv2[QK_PQ2_0] __attribute__((aligned(64)));
                float dr[1] __attribute__((aligned(64)));
                memset(area, 0, 64 + 64 + 256);
                memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
                { uint16_t h; memcpy(&h, area + off, 2); dr[0] = bonsai_fp16_to_fp32(h); }
                for (int jj = 0; jj < QK_PQ2_0; ++jj) {
                    for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
                    xv[jj] = 1.0f;
                    build_xp(xv, xv2, QK_PQ2_0);
                    const float a = v1_row(area + off, xv, QK_PQ2_0);
                    const float b = K[k].f(area + off, xv2, QK_PQ2_0, dr);
                    if (fabsf(a - b) > 1e-6f) ++tot;
                }
                { float xr[QK_PQ2_0] __attribute__((aligned(64)));
                  float xr2[QK_PQ2_0] __attribute__((aligned(64)));
                  for (int t = 0; t < QK_PQ2_0; ++t) xr[t] = x[t];
                  build_xp(xr, xr2, QK_PQ2_0);
                  const float a = v1_row(area + off, xr, QK_PQ2_0);
                  const float b = K[k].f(area + off, xr2, QK_PQ2_0, dr);
                  if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) ++tot; }
            }
        }
        printf("  %-30s MISMATCHES = %d  %s\n", K[k].name, tot,
               tot == 0 ? "*** EXACT ***" : "*** WRONG ***");
    }
    /* multi-block, odd and even nblk */
    printf("\n--- multi-block (4 rows) relative diff: n=512 (4 blocks) and n=640 (5 blocks) ---\n");
    for (int ni = 0; ni < 2; ++ni) {
        const int nn = ni ? 640 : 512;
        float *x2  = (float *)aligned_alloc(64, ((size_t)nn * 4 + 63) & ~(size_t)63);
        float *xp2 = (float *)aligned_alloc(64, ((size_t)nn * 4 + 63) & ~(size_t)63);
        float dr4[8] __attribute__((aligned(64)));
        for (int i = 0; i < nn; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp2, nn);
        for (unsigned k = 0; k < NK; ++k) {
            double worst = 0.0;
            for (int r = 0; r < 4; ++r) {
                const uint8_t *row = w + (size_t)r * row_bytes;
                const int nb = nn / 128;
                for (int b = 0; b < nb; ++b) {
                    uint16_t h; memcpy(&h, row + (size_t)b * 34, 2);
                    dr4[b] = bonsai_fp16_to_fp32(h);
                }
                const double a = v1_row(row, x2, nn);
                const double b = K[k].f(row, xp2, nn, dr4);
                const double e = fabs(a - b) / (fabs(a) + 1.0);
                if (e > worst) worst = e;
            }
            printf("  n=%3d %-30s max rel diff = %.3e  %s\n", nn, K[k].name, worst,
                   worst < 1e-5 ? "OK" : "*** WRONG ***");
        }
        free(x2); free(xp2);
    }

    /* ---- pricing, forward then reversed (order effect check) ----------- */
    double wps[8];
    printf("\n=== PHASE A: kernel only (drow built ONCE outside the timed loop) ===\n");
    for (int pass = 0; pass < 2; ++pass) {
        printf("--- pass %d (single thread, best of %d)%s ---\n",
               pass, reps, pass ? "  [REVERSED ORDER]" : "");
        double sink = 0.0, ref = 0.0;
        build_drow(w, nblk, drow);
        for (unsigned kk = 0; kk < NK; ++kk) {
            const unsigned k = pass ? (NK - 1 - kk) : kk;
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now_s();
                for (int i = 0; i < d; ++i)
                    sink += K[k].f(w + (size_t)i * row_bytes, xp, n, drow);
                const double t1 = now_s();
                if (t1 - t0 < best) best = t1 - t0;
            }
            wps[k] = (double)n * d / best;
            if (k == 0) ref = wps[k];
            if (pass == 0)
                printf("  %-30s %8.1f M w/s   %6.3fx vs V1   %6.3fx vs V10c\n",
                       K[k].name, wps[k] / 1e6, wps[k] / ref, wps[k] / wps[0]);
            else
                printf("  %-30s %8.1f M w/s   %6.3fx vs V10c\n",
                       K[k].name, wps[k] / 1e6, wps[k] / wps[0]);
        }
        printf("  (sink=%g)\n", sink);
        if (pass == 0) {
            /* snapshot of V1 = V10c's number in this binary for the B phase */
        }
    }

    printf("\n=== PHASE B: whole row work, i.e. scale-array build INCLUDED ===\n");
    for (int pass = 0; pass < 2; ++pass) {
        printf("--- pass %d%s ---\n", pass, pass ? "  [REVERSED ORDER]" : "");
        double sink = 0.0, ref = 0.0;
        for (unsigned kk = 0; kk < NK; ++kk) {
            const unsigned k = pass ? (NK - 1 - kk) : kk;
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now_s();
                for (int i = 0; i < d; ++i) {
                    const uint8_t *row = w + (size_t)i * row_bytes;
                    build_drow(row, nblk, drow);
                    sink += K[k].f(row, xp, n, drow);
                }
                const double t1 = now_s();
                if (t1 - t0 < best) best = t1 - t0;
            }
            const double t = (double)n * d / best;
            if (k == 0) ref = t;
            printf("  %-30s %8.1f M w/s   %6.3fx vs V1   %6.3fx vs V10c\n",
                   K[k].name, t / 1e6, t / ref, t / wps[0]);
        }
        printf("  (sink=%g)\n", sink);
    }
    /* true build cost alone */
    {
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            for (int i = 0; i < d; ++i)
                build_drow(w + (size_t)i * row_bytes, nblk, drow);
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        printf("\n  scale-array build alone: %.4f ms for %d rows = %.1f ps/weight "
               "(%.2f M w/s equivalent)\n", best * 1e3, d,
               best * 1e12 / ((double)n * d), (double)n * d / best / 1e6);
    }
    free(w); free(x); free(xp); free(drow); free(area);
    return 0;
}
