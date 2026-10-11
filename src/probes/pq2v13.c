/* pq2v13.c -- cost attribution for the V12 decoder (pq2v12.c), one change per
 * probe, so that the marginal cost of each component is measured in context.
 *
 * V12 is verified exact (0 mismatches, 64 alignments x 129 checks) and priced
 * at 1.330x over the shipped V1 decoder.  This file answers the next question:
 * WHERE do V12's ~129 cycles per 128 weights go?
 *
 * WHY THIS FILE EXISTS: an earlier probe ("P_C" inside pq2v12.c) tried to price
 * one LOAD16 by using only c0 -- but it ALSO replaced the second half's four
 * permutes with plain register copies, so its 22-cycle delta was "1 LOAD16 +
 * 4 permutes", not "1 LOAD16".  Every probe below changes exactly one thing.
 *
 * The V12 skeleton per 128 weights (as emitted by icc; counted by srccount.py):
 *   2 LOAD16 = 2 vmovaps + 2 vloadunpackld + 2 vloadunpackhd
 *   2 vpsrld
 *   2 vmulps   (scale the two LUTs)
 *   8 vpermd   (the LUT gathers = weight materialisation)
 *   8 vfmadd + 4 vmovaps
 *   8 vmovaps  (xp loads) + scalar fp16 decode + loop overhead
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

static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

/* the two LUTs, as values (so no comma-laden macro ever appears in an
 * expression):  LA[j] = (j&3)-1 ,  LB[j] = ((j>>2)&3)-1 */
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

/* the shared tail of every probe: 8 FMA + 8 xp loads into 4 accumulators */
static inline void fma8(__m512 s[4], const __m512 g[8],
                        const float *xp0, const float *xp1) {
    s[0] = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s[0]);
    s[1] = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s[1]);
    s[2] = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s[2]);
    s[3] = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s[3]);
    s[0] = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s[0]);
    s[1] = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s[1]);
    s[2] = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s[2]);
    s[3] = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s[3]);
}
static inline float finish(const __m512 s[4]) {
    __m512 t = _mm512_add_ps(_mm512_add_ps(s[0], s[1]), _mm512_add_ps(s[2], s[3]));
    return _mm512_reduce_add_ps(t);
}

/* ---------------------------------------------------------------------------
 * V12 itself: identical to pq2v12.c, kept here as the reference point.
 * ------------------------------------------------------------------------- */
static float v12_row(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
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
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q1: no per-block fp16 decode (d = constant) ------------------------ */
static float q1_nod(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const __m512 vd = _mm512_set1_ps(1.0009765625f);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q2: the 8 permutes replaced by 8 add_epi32 (same op count) --------- */
static float q2_add(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
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
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_add_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_add_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_add_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_add_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_add_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_add_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_add_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_add_epi32(c1h, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q3: 4 permutes + 4 adds (linearity check) -------------------------- */
static float q3_half(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
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
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_add_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_add_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_add_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_add_epi32(c1h, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q4: ONE LOAD16, eight distinct permutes kept ---------------------- */
static float q4_one16(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);          /* the only LOAD16 */
        const __m512i c1 = _mm512_srli_epi32(c0, 8);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q5: no LUT-scaling multiplies ------------------------------------- */
static float q5_nomul(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512i lau = _mm512_castps_si512(lut_a());
    const __m512i lbu = _mm512_castps_si512(lut_b());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        (void)d;
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lau));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbu));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lau));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbu));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lau));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbu));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lau));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbu));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q6: no shifts at all ---------------------------------------------- */
static float q6_nosrli(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

/* ---- Q7: no xp loads (one hoisted x register) -------------------------- */
static float q7_noxp(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const __m512 xv = _mm512_load_ps(xp);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        s[0] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai)), xv, s[0]);
        s[1] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi)), xv, s[1]);
        s[2] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai)), xv, s[2]);
        s[3] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi)), xv, s[3]);
        s[0] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai)), xv, s[0]);
        s[1] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi)), xv, s[1]);
        s[2] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai)), xv, s[2]);
        s[3] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi)), xv, s[3]);
    }
    return finish(s);
}

/* ---- Q8: 8 permutes + 8 FMA + 8 loads, no weight loads at all ---------- */
static float q8_floor(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512i lai = _mm512_castps_si512(lut_a());
    const __m512i lbi = _mm512_castps_si512(lut_b());
    __m512i r0 = _mm512_set1_epi32((int)((uintptr_t)row & 0xffff));
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512i c0 = r0;
        const __m512i c1 = _mm512_add_epi32(r0, lai);
        const __m512i c0h = _mm512_add_epi32(r0, lbi);
        const __m512i c1h = _mm512_xor_si512(r0, lai);
        s[0] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai)), _mm512_load_ps(xp0 +  0), s[0]);
        s[1] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi)), _mm512_load_ps(xp0 + 16), s[1]);
        s[2] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai)), _mm512_load_ps(xp0 + 32), s[2]);
        s[3] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi)), _mm512_load_ps(xp0 + 48), s[3]);
        s[0] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai)), _mm512_load_ps(xp1 +  0), s[0]);
        s[1] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi)), _mm512_load_ps(xp1 + 16), s[1]);
        s[2] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai)), _mm512_load_ps(xp1 + 32), s[2]);
        s[3] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi)), _mm512_load_ps(xp1 + 48), s[3]);
        r0 = _mm512_add_epi32(r0, c1h);
    }
    return finish(s);
}

/* ---- Q9: 8 FMA + 8 xp loads only (the absolute floor) ------------------ */
static float q9_fmaonly(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 a = lut_a(), bb = lut_b();
    (void)row;
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        s[0] = _mm512_fmadd_ps(a, _mm512_load_ps(xp0 +  0), s[0]);
        s[1] = _mm512_fmadd_ps(bb, _mm512_load_ps(xp0 + 16), s[1]);
        s[2] = _mm512_fmadd_ps(a, _mm512_load_ps(xp0 + 32), s[2]);
        s[3] = _mm512_fmadd_ps(bb, _mm512_load_ps(xp0 + 48), s[3]);
        s[0] = _mm512_fmadd_ps(a, _mm512_load_ps(xp1 +  0), s[0]);
        s[1] = _mm512_fmadd_ps(bb, _mm512_load_ps(xp1 + 16), s[1]);
        s[2] = _mm512_fmadd_ps(a, _mm512_load_ps(xp1 + 32), s[2]);
        s[3] = _mm512_fmadd_ps(bb, _mm512_load_ps(xp1 + 48), s[3]);
    }
    return finish(s);
}

/* ---- Q10: block scale from a vector element instead of scalar fp16 ------
 * What a batched/vectorised fp16 decode would look like: the 16 block scales
 * of a group would be decoded once with vector ops and the per-block scale
 * extracted with one permute.  Here it is faked (wrong values) to price the
 * dependency structure. */
static float q10_vecscale(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const __m512 dv = _mm512_set1_ps(1.0009765625f);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const __m512 vd = _mm512_castsi512_ps(_mm512_permutevar_epi32(
                              _mm512_set1_epi32((int)(b & 15)),
                              _mm512_castps_si512(dv)));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * 128, *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        fma8(s, g, xp0, xp1);
    }
    return finish(s);
}

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
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;
    build_xp(x, xp, n);

    printf("=== pq2v13: cost attribution for the V12 decoder ===\n");
    printf("n=%d rows=%d best of %d, single thread\n\n", n, d, reps);

    double sink = 0.0;
    struct { const char *name; float (*fn)(const uint8_t *, const float *, int); } V[] = {
        { "V12  full candidate (reference)",            v12_row },
        { "Q1   V12 with d=1.0f (no fp16 decode)",      q1_nod },
        { "Q10  V12 with a permuted vector scale",      q10_vecscale },
        { "Q2   V12 with 8 adds instead of 8 permd",    q2_add },
        { "Q3   V12 with 4 permd + 4 adds",             q3_half },
        { "Q4   V12 with 1 LOAD16 (8 permd kept)",      q4_one16 },
        { "Q5   V12 without the 2 LUT-scaling muls",    q5_nomul },
        { "Q6   V12 without any shifts",                q6_nosrli },
        { "Q7   V12 without the xp loads",              q7_noxp },
        { "Q8   8 permd + 8 FMA + 8 loads only",        q8_floor },
        { "Q9   8 FMA + 8 loads, no decode (floor)",    q9_fmaonly },
    };
    const unsigned nv = sizeof V / sizeof V[0];
    double base = 0.0;
    printf("%-44s %10s %8s %10s\n", "variant", "M w/s", "ratio", "cyc/block");
    for (unsigned v = 0; v < nv; ++v) {
        const double wps = price_row(V[v].fn, w, row_bytes, xp, n, d, reps, &sink);
        if (v == 0) base = wps;
        const double cyc = 128.0 / wps * 1.238e9;
        printf("  %-42s %10.1f %8.3f %10.1f\n", V[v].name, wps / 1e6, wps / base, cyc);
    }
    printf("\ncycles/block at 1.238 GHz (measured cpu MHz 1238.094)\n");
    printf("Q1..Q10 are numerically WRONG (deliberate probes); only V12 is exact.\n");
    printf("sink=%.6g\n", sink);
    free(xp); free(w); free(x);
    return 0;
}
