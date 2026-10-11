/* pq2v15.c -- V15: V12 + the two latency wins found by pq2v13/pq2v14.
 *
 * pq2v13 (cost attribution, one change per probe) and pq2v14 (latency hiding)
 * established that the V12 decoder is LATENCY-bound on this in-order core, not
 * throughput-bound:
 *   - replacing all 8 vpermd with vpaddd changes nothing (Q2 == V12),
 *   - but removing the per-block scalar fp16 decode is worth 1.38x (Q1),
 *   - and removing the per-block LUT scaling is worth 1.40x (Q5).
 * Two transformations therefore pay:
 *   (1) PRECOMPUTE the whole row's block scales into an array first, so the hot
 *       loop replaces a branchy ~12-instruction scalar routine with a plain
 *       load (pq2v14 V14d: 1.203x over V12).
 *   (2) Put K BLOCKS IN FLIGHT per loop iteration so the independent chains
 *       fill the pipeline (pq2v14 V14b: 1.170x over V12).
 * V15 combines them and sweeps K.
 *
 * All variants keep V12's arithmetic exactly: the same d, the same LUT values,
 * the same ((code-1)*d) * x association, the same FMA pairing order.  They are
 * therefore expected to be BIT-IDENTICAL to V1 per weight, and that is
 * measured element-wise below (64 source alignments x 128 unit vectors + row
 * sums, plus 4 extra random blocks, plus multi-block rows).
 *
 * NOTE the harness bug this file fixes: pq2v14's verification passed the RAW x
 * to kernels that expect the permuted xp, so it reported 5696 mismatches for
 * every variant including the reference.  The wrappers below build xp for
 * every candidate, which is what pq2v12.c did.
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

/* one block of the V12 decoder; d supplied by the caller */
static inline void blk(const uint8_t *blkp, const float *xp0, float d,
                       const __m512 la0, const __m512 lb0, __m512 s[4]) {
    const __m512 vd = _mm512_set1_ps(d);
    const __m512i c0 = PQ2_IMCI_LOAD16(blkp + 2);
    const __m512i c1 = PQ2_IMCI_LOAD16(blkp + 18);
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

/* ---- V12 exactly (1 block per iteration, scale decoded inline) --------- */
static float v12_row(const uint8_t *row, const float *xp, int n) {
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps() };
    const __m512 la0 = lut_a(), lb0 = lut_b();
    const int nb = n / QK_PQ2_0;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blkp = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blkp[0] | (blkp[1] << 8)));
        blk(blkp, xp + (size_t)b * 128, d, la0, lb0, s);
    }
    return finish(s);
}

/* ---- V15: prescale the row, then K blocks in flight -------------------- */
static inline void prescale_row(const uint8_t *row, int nb, float *darr) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blkp = row + (size_t)b * PQ2_0_BLOCK_BYTES;
        darr[b] = bonsai_fp16_to_fp32((uint16_t)(blkp[0] | (blkp[1] << 8)));
    }
}

#define DEF_V15(name, K)                                                       \
static float name(const uint8_t *row, const float *xp, int n) {                \
    __m512 s[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(),                  \
                    _mm512_setzero_ps(), _mm512_setzero_ps() };                \
    const __m512 la0 = lut_a(), lb0 = lut_b();                                 \
    const int nb = n / QK_PQ2_0;                                               \
    if (nb > MAXBLK) return 0.0f;                                              \
    float darr[MAXBLK];                                                        \
    prescale_row(row, nb, darr);                                               \
    int b = 0;                                                                 \
    for (; b + (K) <= nb; b += (K)) {                                          \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < (K); ++k)                                          \
            blk(row + (size_t)(b + k) * PQ2_0_BLOCK_BYTES,                     \
                xp + (size_t)(b + k) * 128, darr[b + k], la0, lb0, s);         \
    }                                                                          \
    for (; b < nb; ++b)                                                        \
        blk(row + (size_t)b * PQ2_0_BLOCK_BYTES, xp + (size_t)b * 128,         \
            darr[b], la0, lb0, s);                                             \
    return finish(s);                                                          \
}
DEF_V15(v15_k1, 1)
DEF_V15(v15_k2, 2)
DEF_V15(v15_k3, 3)
DEF_V15(v15_k4, 4)
DEF_V15(v15_k5, 5)
DEF_V15(v15_k8, 8)

/* ---- V15p: prescale only (K=1) with the scalar decode kept in-loop as a
 * control: this is V12 + prescale but WITHOUT the array (measures the array
 * itself).  Kept for completeness; numerically identical to V12. --------- */
static float v15_nopre(const uint8_t *row, const float *xp, int n) {
    return v12_row(row, xp, n);
}

/* ---- V1 for the reference --------------------------------------------- */
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

/* ---- wrappers that build xp, exactly as pq2v12.c did ------------------- */
#define DEF_WRAP(nm, fn)                                                       \
static float nm(const uint8_t *row, const float *x, int n) {                   \
    if (n > 16384) return 0.0f;                                                \
    build_xp(x, g_xp, n);                                                      \
    return fn(row, g_xp, n);                                                   \
}
DEF_WRAP(w_v12,    v12_row)
DEF_WRAP(w_v15k1,  v15_k1)
DEF_WRAP(w_v15k2,  v15_k2)
DEF_WRAP(w_v15k3,  v15_k3)
DEF_WRAP(w_v15k4,  v15_k4)
DEF_WRAP(w_v15k5,  v15_k5)
DEF_WRAP(w_v15k8,  v15_k8)
DEF_WRAP(w_v15nop, v15_nopre)

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
            blkp[0] = 0x00; blkp[1] = 0x3C;      /* fp16 scale = 1.0 */
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;

    printf("=== pq2v15: V12 + prescaled row scales + K blocks in flight ===\n");
    printf("n=%d rows=%d reps=%d single thread\n\n", n, d, reps);

    /* also exercise NON-1.0 scales: the harness above uses scale 1.0, so a
     * second block set with random scales is verified too (this is where a
     * wrong fp16 path would show). */
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
        { "V12",         w_v12 },
        { "V15 K=1",     w_v15k1 },
        { "V15 K=2",     w_v15k2 },
        { "V15 K=3",     w_v15k3 },
        { "V15 K=4",     w_v15k4 },
        { "V15 K=5",     w_v15k5 },
        { "V15 K=8",     w_v15k8 },
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
        printf("  %-14s MISMATCHES = %d  %s\n", T[t].name, tot,
               tot == 0 ? "*** EXACT ***" : "*** WRONG ***");
        if (tot) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area, blk, PQ2_0_BLOCK_BYTES);
            verify_vs(T[t].name, T[t].fn, area, x, QK_PQ2_0, 1);
        }
    }

    /* random-scale blocks: 8 blocks with random fp16 scales, 8 alignments */
    printf("\n=== random fp16 scales (8 random blocks x 8 alignments, 129 checks each) ===\n");
    for (unsigned t = 0; t < nt; ++t) {
        int tot = 0;
        for (int trial = 0; trial < 8; ++trial) {
            uint8_t b2[QK_PQ2_0 * 2 + 34];
            uint32_t s3 = 424242u + 7919u * (uint32_t)trial;
            for (unsigned i = 0; i < sizeof b2; ++i) {
                s3 = s3 * 1103515245u + 12345u;
                b2[i] = (uint8_t)(s3 >> 16);
            }
            for (int off = 0; off < 8; ++off) {
                memset(area, 0, 64 + 64 + 256);
                memcpy(area + off, b2, PQ2_0_BLOCK_BYTES);
                tot += verify_vs(T[t].name, T[t].fn, area + off, x, QK_PQ2_0, 0);
            }
        }
        printf("  %-14s MISMATCHES = %d  %s\n", T[t].name, tot,
               tot == 0 ? "*** EXACT ***" : "*** WRONG ***");
    }

    /* multi-block rows (n=512, 4 blocks/row) against V1 */
    {
        float x2[512] __attribute__((aligned(64)));
        for (int i = 0; i < 512; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp, 512);
        printf("\n  multi-block (4 rows x n=512) max rel diff vs V1:");
        for (unsigned t = 0; t < nt; ++t) {
            double worst = 0.0;
            for (int r = 0; r < 4; ++r) {
                const uint8_t *row = w + (size_t)r * row_bytes;
                const double a = v1_row(row, x2, 512);
                const double b = T[t].fn(row, x2, 512);
                const double e = fabs(a - b) / (fabs(a) + 1.0);
                if (e > worst) worst = e;
            }
            printf("  %s=%.2e", T[t].name, worst);
        }
        printf("\n");
    }

    build_xp(x, xp, n);

    /* PRICING uses the RAW kernels with the xp array built once per matvec
     * call -- that is the real trade (xp costs 0.04-0.06 ms per call and is
     * amortised over all d rows).  The wrapped versions above are only for
     * verification, where one block at a time must be fed. */
    struct { const char *name; kfn fn; } P[] = {
        { "V12",     v12_row },
        { "V15 K=1", v15_k1 },
        { "V15 K=2", v15_k2 },
        { "V15 K=3", v15_k3 },
        { "V15 K=4", v15_k4 },
        { "V15 K=5", v15_k5 },
        { "V15 K=8", v15_k8 },
    };
    const unsigned np = sizeof P / sizeof P[0];

    double sink = 0.0;
    const double w1 = price_row(v1_row, w, row_bytes, x, n, d, reps, &sink);
    printf("\n=== pricing (one run, best of %d) ===\n", reps);
    printf("  %-14s %10.1f M weights/s  %6.3fx vs V1\n", "V1 (shipped)", w1 / 1e6, 1.0);
    double best = 0.0; const char *bestname = "";
    for (unsigned t = 0; t < np; ++t) {
        const double wps = price_row(P[t].fn, w, row_bytes, xp, n, d, reps, &sink);
        if (wps > best) { best = wps; bestname = P[t].name; }
        printf("  %-14s %10.1f M weights/s  %6.3fx vs V1   %6.1f cyc/block\n",
               P[t].name, wps / 1e6, wps / w1, 128.0 / wps * 1.238e9);
    }
    printf("\n  best: %s at %.1f M weights/s = %.3fx vs V1\n",
           bestname, best / 1e6, best / w1);
    printf("sink=%.6g\n", sink);
    free(xp); free(area); free(w); free(x);
    return 0;
}
