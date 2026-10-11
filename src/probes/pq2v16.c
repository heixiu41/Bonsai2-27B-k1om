/* pq2v16.c -- controlled head-to-head benchmark and the layout question.
 *
 * A PROBLEM THAT MUST BE SETTLED FIRST
 * ------------------------------------
 * pq2v12.c and pq2v14.c measured the SAME v12_row source at 1216-1228 M w/s;
 * pq2v15.c measured it at 1461 M w/s, and its V1 reference at 718 M w/s where
 * pq2v12/pq2v14 measured 923-934.  Same source, same card, same clock
 * (cpu MHz 1238.094), load average 0.00.  So on this in-order core the
 * COMPILER'S SCHEDULING AND CODE PLACEMENT dominate the absolute numbers, and
 * a ratio taken from a binary that contains only one of the two kernels is
 * not trustworthy.
 *
 * This file therefore measures every kernel in ONE binary, and runs the whole
 * price list TWICE (forward and reversed) so that any order effect is visible.
 * It also separates two changes that pq2v15 accidentally bundled:
 *   - writing the block body inline vs. through a shared static-inline helper
 *     (which changes what the scheduler sees),
 *   - prescaling the row's block scales into an array first.
 * The matrix is 2 x 2 (inline/helper) x (inline-decode/prescaled).
 *
 * All six kernels below are the same arithmetic (V12's scale-in-LUT + two
 * LUTs), so ALL of them must verify exactly; the only differences are
 * scheduling and where the block scale comes from.
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

#define MAXBLK 4096
static float g_xp[16384] __attribute__((aligned(64)));

static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

static inline __m512 lut_a(void) {
    return _mm512_castsi512_ps(_mm512_set_epi32(
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u));
}
static inline __m512 lut_b(void) {
    return _mm512_castsi512_ps(_mm512_set_epi32(
        (int)0x40000000u,(int)0x40000000u,(int)0x40000000u,(int)0x40000000u,
        (int)0x3F800000u,(int)0x3F800000u,(int)0x3F800000u,(int)0x3F800000u,
        0,0,0,0,
        (int)0xBF800000u,(int)0xBF800000u,(int)0xBF800000u,(int)0xBF800000u));
}
static inline float finish(const __m512 s[4]) {
    __m512 t = _mm512_add_ps(_mm512_add_ps(s[0], s[1]), _mm512_add_ps(s[2], s[3]));
    return _mm512_reduce_add_ps(t);
}

/* ============ 1. V1: the shipped decoder (reference) ==================== */
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

/* ============ 2. V10c: the previous agent's verified best =============== */
static float v10c_row(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
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

/* ============ 3. V12 written out INLINE (exactly as pq2v12.c) =========== */
static float v12_inline(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 la0 = lut_a(), lb0 = lut_b();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la), lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
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

/* ============ 4. V12 via a shared helper ================================ */
static inline void blk_helper(const uint8_t *blkp, const float *xp0, float d,
                              const __m512 la0, const __m512 lb0, __m512 s[4]) {
    const __m512 vd = _mm512_set1_ps(d);
    const __m512i c0 = PQ2_IMCI_LOAD16(blkp + 2);
    const __m512i c1 = PQ2_IMCI_LOAD16(blkp + 18);
    const __m512i c0h = _mm512_srli_epi32(c0, 4);
    const __m512i c1h = _mm512_srli_epi32(c1, 4);
    const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
    const __m512i lai = _mm512_castps_si512(la), lbi = _mm512_castps_si512(lb);
    const float *xp1 = xp0 + 64;
    __m512 g[8];
    g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
    g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
    g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
    g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
    g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
    g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
    g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
    g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
    s[0] = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s[0]);
    s[1] = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s[1]);
    s[2] = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s[2]);
    s[3] = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s[3]);
    s[0] = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s[0]);
    s[1] = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s[1]);
    s[2] = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s[2]);
    s[3] = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s[3]);
}
static float v12_helper(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const int nb = n / QK_PQ2_0;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blkp = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blkp[0] | (blkp[1] << 8)));
        blk_helper(blkp, xp + (size_t)b * 128, d, la0, lb0, s);
    }
    return finish(s);
}

/* ============ 5. helper + prescaled row scales ========================== */
static inline void prescale_row(const uint8_t *row, int nb, float *darr) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blkp = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        darr[b] = bonsai_fp16_to_fp32((uint16_t)(blkp[0] | (blkp[1] << 8)));
    }
}
static float v12_helper_pre(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const int nb = n / QK_PQ2_0;
    if (nb > MAXBLK) return 0.0f;
    float darr[MAXBLK];
    prescale_row(row, nb, darr);
    for (int b = 0; b < nb; ++b)
        blk_helper(row + (size_t)b * PQ2_0_BLOCK_BYTES, xp + (size_t)b * 128,
                   darr[b], la0, lb0, s);
    return finish(s);
}

/* ============ 6. inline + prescaled row scales ========================== */
static float v12_inline_pre(const uint8_t *row, const float *xp, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const int nb = n / QK_PQ2_0;
    if (nb > MAXBLK) return 0.0f;
    float darr[MAXBLK];
    prescale_row(row, nb, darr);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        const float d = darr[b];
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + (size_t)b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la), lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
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

/* ============ 7. inline + EXPLICIT software prefetch of the weight row ===
 * The helper version's disassembly contains vprefetch0/vprefetch1 that the
 * inline version lacks, which is the likely reason it is 19% faster.  If an
 * explicit prefetch reproduces the win, the effect is robust rather than an
 * accident of how the source happened to be factored.
 * ======================================================================== */
#define DEF_PF(name, DIST)                                                     \
static float name(const uint8_t *row, const float *xp, int n) {                \
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();                 \
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();                 \
    const __m512 la0 = lut_a(), lb0 = lut_b();                                 \
    const int nb = n / QK_PQ2_0;                                               \
    for (int b = 0; b < nb; ++b) {                                             \
        if (b + (DIST) < nb)                                                   \
            _mm_prefetch((const char *)(row + (size_t)(b + (DIST)) *            \
                                        PQ2_0_BLOCK_BYTES), _MM_HINT_T0);       \
        const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;              \
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));\
        const __m512 vd = _mm512_set1_ps(d);                                   \
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);                           \
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);                          \
        const __m512i c0h = _mm512_srli_epi32(c0, 4);                          \
        const __m512i c1h = _mm512_srli_epi32(c1, 4);                          \
        const float *xp0 = xp + (size_t)b * 128, *xp1 = xp0 + 64;              \
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd); \
        const __m512i lai = _mm512_castps_si512(la);                           \
        const __m512i lbi = _mm512_castps_si512(lb);                           \
        __m512 g[8];                                                           \
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));         \
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));         \
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));         \
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));         \
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));         \
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));         \
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));         \
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));         \
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);              \
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);              \
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);              \
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);              \
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);              \
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);              \
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);              \
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);              \
    }                                                                          \
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));    \
    return _mm512_reduce_add_ps(s);                                            \
}
DEF_PF(v12_pf4,  4)
DEF_PF(v12_pf8,  8)
DEF_PF(v12_pf16, 16)
DEF_PF(v12_pf32, 32)

/* ---- wrappers for verification (build xp, pq2v12-style) ---------------- */
#define DEF_WRAP(nm, fn)                                                       \
static float nm(const uint8_t *row, const float *x, int n) {                   \
    if (n > 16384) return 0.0f;                                                \
    build_xp(x, g_xp, n);                                                      \
    return fn(row, g_xp, n);                                                   \
}
DEF_WRAP(w_v12i,  v12_inline)
DEF_WRAP(w_v12h,  v12_helper)
DEF_WRAP(w_v12hp, v12_helper_pre)
DEF_WRAP(w_v12ip, v12_inline_pre)
DEF_WRAP(w_v10c,  v10c_row)

typedef float (*kfn)(const uint8_t *, const float *, int);

static int verify_vs(const char *label, kfn cand, const uint8_t *area,
                     const float *x, int n, int verbose) {
    int bad = 0;
    float xv[QK_PQ2_0] __attribute__((aligned(64)));
    for (int jj = 0; jj < QK_PQ2_0; ++jj) {
        for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
        xv[jj] = 1.0f;
        const float a = v1_row(area, xv, QK_PQ2_0);
        const float b = cand(area, xv, QK_PQ2_0);
        if (fabsf(a - b) > 1e-6f) {
            if (verbose && bad < 6)
                printf("      %s unit-vector mismatch at weight %d: V1=%.7g cand=%.7g\n",
                       label, jj, a, b);
            ++bad;
        }
    }
    const float a = v1_row(area, x, n);
    const float b = cand(area, x, n);
    if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) {
        if (verbose) printf("      %s row-sum mismatch: V1=%.7g cand=%.7g\n", label, a, b);
        ++bad;
    }
    return bad;
}

static double price_row(kfn fn, const uint8_t *w, size_t row_bytes,
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
    setvbuf(stdout, NULL, _IONBF, 0);
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
            uint8_t *blkp = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            blkp[0] = 0x00; blkp[1] = 0x3C;
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;

    printf("=== pq2v16: controlled head-to-head (one binary, two passes) ===\n");
    printf("n=%d rows=%d reps=%d single thread\n\n", n, d, reps);

    /* verification: 64 alignments x 129 checks, then 8 random-scale blocks */
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

    struct { const char *name; kfn fn; } T[] = {
        { "V10c",       w_v10c },
        { "V12 inline", w_v12i },
        { "V12 helper", w_v12h },
        { "V12 inline+pre", w_v12ip },
        { "V12 helper+pre", w_v12hp },
    };
    const unsigned nt = sizeof T / sizeof T[0];
    printf("=== element-wise vs V1 (64 alignments x 129 checks + 8 random-scale blocks) ===\n");
    for (unsigned t = 0; t < nt; ++t) {
        int tot = 0;
        for (int off = 0; off < 64; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
            tot += verify_vs(T[t].name, T[t].fn, area + off, x, QK_PQ2_0, 0);
        }
        /* Random scales that are FINITE and mostly normal.  A purely random
         * 16-bit value is usually NaN/Inf, and NaN==NaN comparisons pass
         * vacuously (fabsf(NaN) > tol is false), so a random-half test would
         * be worthless.  These span normal, subnormal and signed values. */
        for (int trial = 0; trial < 8; ++trial) {
            uint8_t b2[QK_PQ2_0 * 2 + 34];
            uint32_t s3 = 424242u + 7919u * (uint32_t)trial;
            for (unsigned i = 0; i < sizeof b2; ++i) {
                s3 = s3 * 1103515245u + 12345u;
                b2[i] = (uint8_t)(s3 >> 16);
            }
            /* finite half: sign | exponent in [1,30] | mantissa */
            const unsigned ex = 1u + (s3 % 30u);
            const unsigned h = ((s3 >> 8) & 0x8000u) | (ex << 10) | (s3 & 0x3FFu);
            b2[0] = (uint8_t)(h & 0xFF);
            b2[1] = (uint8_t)(h >> 8);
            for (int off = 0; off < 8; ++off) {
                memset(area, 0, 64 + 64 + 256);
                memcpy(area + off, b2, PQ2_0_BLOCK_BYTES);
                tot += verify_vs(T[t].name, T[t].fn, area + off, x, QK_PQ2_0, 0);
            }
        }
        /* explicit edge-case scales */
        {
            static const uint16_t hs[] = { 0x0000u, 0x8000u, 0x0001u, 0x03FFu,
                                           0x0400u, 0x7BFFu, 0xC000u, 0x3C00u,
                                           0x007Fu, 0xFBFFu };
            for (unsigned e = 0; e < sizeof hs / sizeof hs[0]; ++e) {
                uint8_t b2[QK_PQ2_0 * 2 + 34];
                uint32_t s3 = 777u + 13u * e;
                for (unsigned i = 0; i < sizeof b2; ++i) {
                    s3 = s3 * 1103515245u + 12345u;
                    b2[i] = (uint8_t)(s3 >> 16);
                }
                b2[0] = (uint8_t)(hs[e] & 0xFF);
                b2[1] = (uint8_t)(hs[e] >> 8);
                for (int off = 0; off < 8; ++off) {
                    memset(area, 0, 64 + 64 + 256);
                    memcpy(area + off, b2, PQ2_0_BLOCK_BYTES);
                    tot += verify_vs(T[t].name, T[t].fn, area + off, x, QK_PQ2_0, 0);
                }
            }
        }
        printf("  %-18s total checks = %d, MISMATCHES = %d  %s\n",
               T[t].name, 64 * 129 + (8 + 10) * 8 * 129, tot,
               tot == 0 ? "*** EXACT ***" : "*** WRONG ***");
        if (tot) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area, blk, PQ2_0_BLOCK_BYTES);
            verify_vs(T[t].name, T[t].fn, area, x, QK_PQ2_0, 1);
        }
    }

    build_xp(x, xp, n);

    struct { const char *name; kfn fn; } P[] = {
        { "V1 shipped",      v1_row },
        { "V10c",            v10c_row },
        { "V12 inline",      v12_inline },
        { "V12 helper",      v12_helper },
        { "V12 inline+pre",  v12_inline_pre },
        { "V12 helper+pre",  v12_helper_pre },
        { "V12 pf4",         v12_pf4 },
        { "V12 pf8",         v12_pf8 },
        { "V12 pf16",        v12_pf16 },
        { "V12 pf32",        v12_pf32 },
    };
    const unsigned np = sizeof P / sizeof P[0];
    static double res[2][10];
    double sink = 0.0;

    for (int pass = 0; pass < 2; ++pass) {
        printf("\n=== pricing pass %d (%s order, best of %d) ===\n",
               pass + 1, pass == 0 ? "forward" : "REVERSE", reps);
        for (unsigned k = 0; k < np; ++k) {
            const unsigned t = pass == 0 ? k : (np - 1 - k);
            const float *xv = (t == 0) ? x : xp;
            res[pass][t] = price_row(P[t].fn, w, row_bytes, xv, n, d, reps, &sink);
        }
        for (unsigned t = 0; t < np; ++t)
            printf("  %-18s %10.1f M weights/s  %6.3fx vs V1\n",
                   P[t].name, res[pass][t] / 1e6, res[pass][t] / res[pass][0]);
    }
    printf("\n=== summary (ratio to V1 in the SAME pass) ===\n");
    printf("  %-18s %8s %8s\n", "kernel", "pass1", "pass2");
    for (unsigned t = 0; t < np; ++t)
        printf("  %-18s %8.3f %8.3f\n", P[t].name,
               res[0][t] / res[0][0], res[1][t] / res[1][0]);
    printf("\nsink=%.6g\n", sink);
    free(xp); free(area); free(w); free(x);
    return 0;
}
