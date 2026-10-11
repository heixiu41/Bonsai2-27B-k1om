/* pq2v14.c -- latency hiding for the V12 decoder.
 *
 * WHAT pq2v13 ESTABLISHED (all measured on the card, one change per probe):
 *
 *   V12 full candidate ............................. 1229.5 M w/s  128.9 cyc/block
 *   Q1  no per-block fp16 decode (d const) ......... 1691.4        93.7   +37%
 *   Q5  no LUT-scaling multiplies .................. 1723.2        92.0   +40%
 *   Q10 block scale from a vector (no scalar decode) 1559.7       101.6   +27%
 *   Q2  8 vpermd replaced by 8 vpaddd .............. 1228.6       129.0   +0%
 *   Q4  only ONE LOAD16 instead of two ............. 1273.2       124.5   +3.6%
 *   Q6  no shifts ................................. 1412.4       112.2   +15%
 *   Q7  no xp loads ............................... 1337.9       118.4   +8.8%
 *   Q8  8 permutes + 8 FMA + 8 loads only .......... 3127.6        50.7   +154%
 *   Q9  8 FMA + 8 loads only (absolute floor) ...... 4864.7        32.6   +296%
 *
 * TWO CONCLUSIONS, BOTH SURPRISING:
 *  (1) vpermd is NOT expensive.  Q2 (8 permutes -> 8 adds, same op count) is
 *      EXACTLY the same speed.  The "vpermd is 62% of the time" premise is
 *      dead: every 512-bit ALU op costs the same, so only the OP COUNT
 *      matters.  (An earlier probe in pq2v12.c appeared to contradict this --
 *      it had silently removed the LUT-scaling multiplies too, which is worth
 *      1.40x, so it was confounded.)
 *  (2) The dominant cost is the per-block DEPENDENCY CHAIN through
 *      load -> scalar fp16 decode -> broadcast -> 2 LUT multiplies -> permutes
 *      -> FMA, on an in-order core whose compiler is not software-pipelining
 *      the block loop.  Breaking that chain (Q1/Q5) is worth 1.38-1.40x, far
 *      more than any instruction-count change.
 *
 * This file tests four ways to hide or shorten that latency, on top of V12:
 *   A/B  unroll the block loop by 2 (two independent blocks in flight)
 *   C    scale the x operand instead of the LUT (the permutes stop depending
 *        on d; costs 8 multiplies instead of 2 but they are off the chain)
 *   D    decode ALL 40 block scales for the row into an array first, so the
 *        inner loop sees a plain load instead of a branchy scalar routine
 *   E    C + D combined
 * Exactness is re-measured element-wise for every exact variant; C changes the
 * rounding order of one multiply ((code-1)*(d*x) instead of ((code-1)*d)*x) so
 * it is measured against the tolerance harness AND reported separately.
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

/* ---- one block, LUT-scaled (V12 scheme), scale given as a float --------- */
static inline void blk_lutscale(const uint8_t *blk, const float *xp0, float d,
                                const __m512 la0, const __m512 lb0, __m512 s[4]) {
    const __m512 vd = _mm512_set1_ps(d);
    const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
    const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
    const __m512i c0h = _mm512_srli_epi32(c0, 4);
    const __m512i c1h = _mm512_srli_epi32(c1, 4);
    const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
    const __m512i lai = _mm512_castps_si512(la);
    const __m512i lbi = _mm512_castps_si512(lb);
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

/* ---- one block, x-side scaling: the permutes do NOT depend on d --------- */
static inline void blk_xscale(const uint8_t *blk, const float *xp0, __m512 vd,
                              const __m512i lai, const __m512i lbi, __m512 s[4]) {
    const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
    const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
    const __m512i c0h = _mm512_srli_epi32(c0, 4);
    const __m512i c1h = _mm512_srli_epi32(c1, 4);
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
    s[0] = _mm512_fmadd_ps(g[0], _mm512_mul_ps(_mm512_load_ps(xp0 +  0), vd), s[0]);
    s[1] = _mm512_fmadd_ps(g[1], _mm512_mul_ps(_mm512_load_ps(xp0 + 16), vd), s[1]);
    s[2] = _mm512_fmadd_ps(g[2], _mm512_mul_ps(_mm512_load_ps(xp0 + 32), vd), s[2]);
    s[3] = _mm512_fmadd_ps(g[3], _mm512_mul_ps(_mm512_load_ps(xp0 + 48), vd), s[3]);
    s[0] = _mm512_fmadd_ps(g[4], _mm512_mul_ps(_mm512_load_ps(xp1 +  0), vd), s[0]);
    s[1] = _mm512_fmadd_ps(g[5], _mm512_mul_ps(_mm512_load_ps(xp1 + 16), vd), s[1]);
    s[2] = _mm512_fmadd_ps(g[6], _mm512_mul_ps(_mm512_load_ps(xp1 + 32), vd), s[2]);
    s[3] = _mm512_fmadd_ps(g[7], _mm512_mul_ps(_mm512_load_ps(xp1 + 48), vd), s[3]);
}

/* ===================== A = V12 (reference) =============================== */
static float v12_row(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        blk_lutscale(blk, xp + b * 128, d, la0, lb0, s);
    }
    return finish(s);
}

/* ===================== B = V12, two blocks in flight ==================== */
static float v14b_2blk(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    int j = 0;
    for (; j + 2 * QK_PQ2_0 <= n; j += 2 * QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk0 = row + b * PQ2_0_BLOCK_BYTES;
        const uint8_t *blk1 = blk0 + PQ2_0_BLOCK_BYTES;
        const float d0 = bonsai_fp16_to_fp32((uint16_t)(blk0[0] | (blk0[1] << 8)));
        const float d1 = bonsai_fp16_to_fp32((uint16_t)(blk1[0] | (blk1[1] << 8)));
        blk_lutscale(blk0, xp + b * 128, d0, la0, lb0, s);
        blk_lutscale(blk1, xp + b * 128 + 128, d1, la0, lb0, s);
    }
    for (; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        blk_lutscale(blk, xp + b * 128, d, la0, lb0, s);
    }
    return finish(s);
}

/* ===================== C = x-side scaling =============================== */
static float v14c_xscale(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512i lai = _mm512_castps_si512(lut_a());
    const __m512i lbi = _mm512_castps_si512(lut_b());
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        blk_xscale(blk, xp + b * 128, _mm512_set1_ps(d), lai, lbi, s);
    }
    return finish(s);
}

/* ===================== D = precomputed scales ===========================
 * Decode all block scales for the row first (a separate, nicely pipelined
 * scalar loop), then the hot loop only loads them.  Exact: same value d, same
 * multiply order as V12.
 * ======================================================================== */
#define MAXBLK 4096
static float v14d_prescale(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const int nb = n / QK_PQ2_0;
    float darr[MAXBLK];
    if (nb > MAXBLK) return 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        darr[b] = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk_lutscale(blk, xp + (size_t)b * 128, darr[b], la0, lb0, s);
    }
    return finish(s);
}

/* ===================== E = x-side scaling + precomputed scales ========== */
static float v14e_xprescale(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512i lai = _mm512_castsi512_ps(lut_a());
    const __m512i lbi = _mm512_castsi512_ps(lut_b());
    const int nb = n / QK_PQ2_0;
    float darr[MAXBLK];
    if (nb > MAXBLK) return 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        darr[b] = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk_xscale(blk, xp + (size_t)b * 128, _mm512_set1_ps(darr[b]), lai, lbi, s);
    }
    return finish(s);
}

/* ===================== F = x-side + 2 blocks in flight ================== */
static float v14f_x2blk(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512i lai = _mm512_castsi512_ps(lut_a());
    const __m512i lbi = _mm512_castsi512_ps(lut_b());
    int j = 0;
    for (; j + 2 * QK_PQ2_0 <= n; j += 2 * QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk0 = row + b * PQ2_0_BLOCK_BYTES;
        const uint8_t *blk1 = blk0 + PQ2_0_BLOCK_BYTES;
        const float d0 = bonsai_fp16_to_fp32((uint16_t)(blk0[0] | (blk0[1] << 8)));
        const float d1 = bonsai_fp16_to_fp32((uint16_t)(blk1[0] | (blk1[1] << 8)));
        blk_xscale(blk0, xp + b * 128,       _mm512_set1_ps(d0), lai, lbi, s);
        blk_xscale(blk1, xp + b * 128 + 128, _mm512_set1_ps(d1), lai, lbi, s);
    }
    for (; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        blk_xscale(blk, xp + b * 128, _mm512_set1_ps(d), lai, lbi, s);
    }
    return finish(s);
}

/* ===================== verification harness ============================= */
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
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;

    printf("=== pq2v14: latency hiding on top of V12 ===\n");
    printf("n=%d rows=%d reps=%d single thread\n\n", n, d, reps);

    /* ---- element-wise verification: 64 alignments x (128 unit vec + row) -- */
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
    struct { const char *name; kfn fn; int exact_math; } T[] = {
        { "V12  (LUT-scaled, 1 block/iter)", v12_row,      1 },
        { "V14b (LUT-scaled, 2 blocks/iter)", v14b_2blk,   1 },
        { "V14c (x-scaled, 1 block/iter)",   v14c_xscale,  0 },
        { "V14d (precomputed scales)",       v14d_prescale, 1 },
        { "V14e (x-scaled + prescaled)",     v14e_xprescale, 0 },
        { "V14f (x-scaled + 2 blocks)",      v14f_x2blk,   0 },
    };
    const unsigned nt = sizeof T / sizeof T[0];
    printf("=== element-wise vs V1: 64 alignments x (128 unit vectors + row sum) ===\n");
    for (unsigned t = 0; t < nt; ++t) {
        int tot = 0;
        for (int off = 0; off < 64; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
            tot += verify_vs(T[t].name, T[t].fn, area + off, x, QK_PQ2_0, 0);
        }
        printf("  %-36s MISMATCHES = %d  %s%s\n", T[t].name, tot,
               tot == 0 ? "*** EXACT ***" : "*** WRONG ***",
               T[t].exact_math ? "" : "   (x-scaled: 1e-7 rounding-order change)");
        if (tot) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area, blk, PQ2_0_BLOCK_BYTES);
            verify_vs(T[t].name, T[t].fn, area, x, QK_PQ2_0, 1);
        }
    }
    printf("\n");

    /* extra random blocks x 8 alignments */
    for (int trial = 0; trial < 4; ++trial) {
        uint8_t b2[QK_PQ2_0 * 2 + 34];
        uint32_t s3 = 1234567u + 7919u * (uint32_t)trial;
        for (unsigned i = 0; i < sizeof b2; ++i) {
            s3 = s3 * 1103515245u + 12345u;
            b2[i] = (uint8_t)(s3 >> 16);
        }
        b2[0] = 0x00; b2[1] = 0x3C;
        int bad[6] = {0,0,0,0,0,0};
        for (int off = 0; off < 8; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, b2, PQ2_0_BLOCK_BYTES);
            for (unsigned t = 0; t < nt; ++t)
                bad[t] += verify_vs(T[t].name, T[t].fn, area + off, x, QK_PQ2_0, 0);
        }
        printf("  extra block %d (8 alignments):", trial);
        for (unsigned t = 0; t < nt; ++t) printf(" %s=%d", T[t].name + 0, bad[t]);
        printf("\n");
    }

    /* multi-block rows, vs V1, max relative difference */
    {
        float x2[512] __attribute__((aligned(64)));
        float xp2[512] __attribute__((aligned(64)));
        for (int i = 0; i < 512; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp2, 512);
        printf("\n  multi-block (4 rows x n=512), max rel diff vs V1:\n");
        for (unsigned t = 0; t < nt; ++t) {
            double worst = 0.0;
            for (int r = 0; r < 4; ++r) {
                const uint8_t *row = w + (size_t)r * row_bytes;
                const double a = v1_row(row, x, 512);
                const double b = T[t].fn(row, xp2, 512);
                const double e = fabs(a - b) / (fabs(a) + 1.0);
                if (e > worst) worst = e;
            }
            printf("    %-36s %.3e\n", T[t].name, worst);
        }
    }
    printf("\n");

    build_xp(x, xp, n);

    double sink = 0.0;
    printf("=== pricing (one run, best of %d) ===\n", reps);
    double base = 0.0;
    printf("%-40s %10s %8s %10s\n", "variant", "M w/s", "ratio", "cyc/block");
    for (unsigned t = 0; t < nt; ++t) {
        const double wps = price_row(T[t].fn, w, row_bytes, xp, n, d, reps, &sink);
        if (t == 0) base = wps;
        printf("  %-38s %10.1f %8.3f %10.1f\n", T[t].name, wps / 1e6, wps / base,
               128.0 / wps * 1.238e9);
    }
    printf("\nsink=%.6g\n", sink);
    free(xp); free(area); free(w); free(x);
    return 0;
}
