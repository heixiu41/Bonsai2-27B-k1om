/* pq2f3batch.c -- does batching help the F3 kernel at 244 threads?
 *
 * WHY THIS PROBE EXISTS: the old pq2batch.c measured the V1 decoder
 * (~36 decode ops/block) with a single serial 8-deep FMA chain plus a
 * per-(block,token) reduce_add, default 1 thread. Its "B=16 -> 1.11x"
 * says nothing about the CURRENT F3 kernel (14 decode ops/block) at the
 * production thread count. This probe answers that, decisively:
 *
 *   CONTROL : the shipped pq2_0_matvec_knc called B times (once per token).
 *   BATCHED : one pass over the weights; each 128-weight block is decoded
 *             ONCE (identical F3 sequence: 2 LOAD16 + 2 srli + 2 LUT muls
 *             + 8 permutevar), then each of the B tokens runs the SAME FMA
 *             chain the shipped kernel runs. Per-token arithmetic order is
 *             unchanged, so batched output must be BIT-IDENTICAL to the
 *             control (verified by memcmp below -- the hard gate).
 *
 * The xp permutation builds are INSIDE the timed region for the batched
 * path too (the model would pay them per token either way).
 *
 * B is compile-time specialised (1/2/4/8/16): a variable-length inner loop
 * would spill the accumulators and measure the compiler, not the machine.
 * Register pressure at large B is a REAL cost of batching on KNC (32 zmm
 * registers) and is meant to show up here.
 *
 * Traps avoided (this project's list): a weight byte is flipped before every
 * pass; results feed a printed volatile sink; control/batched order alternates
 * between reps; buffers allocated once; best-of-N.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#include "pq2_0.h"

static volatile float sink;
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- batched F3 matvec, specialised for compile-time B -----------------
 * y[t][i] = W[i] . xp[t]  for t in [0,B). xp[t] is the permuted activation
 * of token t (pq2_build_xp layout), exactly what the shipped kernel consumes.
 * Per-token FMA order is identical to pq2_0_matvec_knc -> bit-identical. */
#define DEF_BATCH(K)                                                        \
static void f3_batch_##K(float *y, const float *xp,                         \
                         const uint8_t *w, int n, int d) {                  \
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;                               \
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;           \
    const __m512 l0b = pq2_lut_a(), l1b = pq2_lut_b();                      \
    _Pragma("omp parallel for schedule(static)")                            \
    for (int i = 0; i < d; ++i) {                                           \
        float drow[PQ2_XP_MAX / QK_PQ2_0] __attribute__((aligned(64)));     \
        const uint8_t *row = w + (size_t)i * rb;                            \
        pq2_build_drow(row, nn / QK_PQ2_0, drow);                           \
        __m512 s[K];                                                        \
        for (int t = 0; t < K; ++t) s[t] = _mm512_setzero_ps();             \
        const uint8_t *blk = row;                                           \
        const float *dp = drow;                                             \
        size_t off = 0;                                                     \
        int nb = nn >> 7;                                                   \
        do {                                                                \
            const __m512 vd = _mm512_set1_ps(*dp);                          \
            const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd)); \
            const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd)); \
            const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);                    \
            const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);                   \
            const __m512i e0 = _mm512_srli_epi32(c0, 4);                    \
            const __m512i e1 = _mm512_srli_epi32(c1, 4);                    \
            __m512 g[8];                                                    \
            g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0));    \
            g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1));    \
            g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0));    \
            g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1));    \
            g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0));    \
            g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1));    \
            g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0));    \
            g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1));    \
            for (int t = 0; t < K; ++t) {                                   \
                const float *x0 = xp + (size_t)t * n + off;                 \
                __m512 u = _mm512_mul_ps(g[0], _mm512_load_ps(x0));         \
                u = _mm512_fmadd_ps(g[1], _mm512_load_ps(x0 +  16), u);     \
                u = _mm512_fmadd_ps(g[2], _mm512_load_ps(x0 +  32), u);     \
                u = _mm512_fmadd_ps(g[3], _mm512_load_ps(x0 +  48), u);     \
                u = _mm512_fmadd_ps(g[4], _mm512_load_ps(x0 +  64), u);     \
                u = _mm512_fmadd_ps(g[5], _mm512_load_ps(x0 +  80), u);     \
                u = _mm512_fmadd_ps(g[6], _mm512_load_ps(x0 +  96), u);     \
                u = _mm512_fmadd_ps(g[7], _mm512_load_ps(x0 + 112), u);     \
                s[t] = _mm512_add_ps(s[t], u);                              \
            }                                                               \
            blk += PQ2_0_BLOCK_BYTES;                                       \
            off += QK_PQ2_0;                                                \
            ++dp;                                                           \
        } while (--nb);                                                     \
        for (int t = 0; t < K; ++t)                                         \
            y[(size_t)t * d + i] = _mm512_reduce_add_ps(s[t]);              \
    }                                                                       \
}

DEF_BATCH(1)
DEF_BATCH(2)
DEF_BATCH(4)
DEF_BATCH(8)
DEF_BATCH(16)

typedef void (*batch_fn)(float *, const float *, const uint8_t *, int, int);
static batch_fn pick(int B) {
    switch (B) {
        case 1: return f3_batch_1;  case 2: return f3_batch_2;
        case 4: return f3_batch_4;  case 8: return f3_batch_8;
        default: return f3_batch_16;
    }
}

/* the model's real folded shapes, in call order */
typedef struct { const char *nm; int n, d; } shp;
static const shp T[] = {
    { "ffn_gate",   5120, 17408 },
    { "ffn_up",     5120, 17408 },
    { "ffn_down",  17408,  5120 },
    { "attn_qkv",   5120, 10240 },
    { "attn_gate",  5120,  6144  },
};
#define NT ((int)(sizeof(T)/sizeof(T[0])))

static uint8_t *mkweights(int n, int d) {
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t bytes = rb * (size_t)d;
    uint8_t *w = (uint8_t *)malloc(bytes);
    if (!w) { fprintf(stderr, "malloc failed\n"); exit(1); }
    uint32_t st = 12345u;
    for (size_t i = 0; i < bytes; ++i) { st = st*1103515245u+12345u; w[i] = (uint8_t)(st>>16); }
    for (int r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + (size_t)r*rb + (size_t)b*PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;      /* fp16 scale = 1.0 */
        }
    return w;
}

int main(int argc, char **argv) {
    const int nth  = argc > 1 ? atoi(argv[1]) : 244;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    omp_set_num_threads(nth);

    printf("=== F3 batched decode: B tokens share one weight decode ===\n");
    printf("threads=%d reps=%d\n\n", nth, reps);

    uint8_t *W[NT];
    for (int t = 0; t < NT; ++t) W[t] = mkweights(T[t].n, T[t].d);

    #define BMAX 16
    #define NMAX 17408
    #define DMAX 17408
    float *x  = (float *)aligned_alloc(64, (size_t)BMAX * NMAX * sizeof(float));
    float *xp = (float *)aligned_alloc(64, (size_t)BMAX * NMAX * sizeof(float));
    float *y  = (float *)aligned_alloc(64, (size_t)BMAX * DMAX * sizeof(float));
    float *yr = (float *)aligned_alloc(64, (size_t)DMAX * sizeof(float));
    if (!x || !xp || !y || !yr) return 1;
    for (int i = 0; i < BMAX * NMAX; ++i)
        x[i] = (float)(((i * 7 + i / 128) % 13) - 6) * 0.125f;

    /* ---- hard gate: batched must be BIT-IDENTICAL to the shipped kernel --
     * same per-token FMA order by construction; verify, don't assume. */
    int bad = 0;
    for (int t = 0; t < NT; ++t) {
        for (int b = 0; b < 4; ++b)
            pq2_build_xp(x + (size_t)b * T[t].n, xp + (size_t)b * T[t].n, T[t].n);
        f3_batch_4(y, xp, W[t], T[t].n, T[t].d);
        for (int b = 0; b < 4; ++b) {
            pq2_0_matvec_knc(yr, x + (size_t)b * T[t].n, W[t], T[t].n, T[t].d);
            if (memcmp(y + (size_t)b * T[t].d, yr,
                       (size_t)T[t].d * sizeof(float)) != 0) {
                ++bad;
                printf("GATE FAIL: %s token %d differs from shipped kernel\n",
                       T[t].nm, b);
            }
        }
    }
    printf("bit-identity gate (B=4, %d shapes): %s\n\n", NT, bad ? "FAILED" : "0 diffs");
    if (bad) return 1;

    /* ---- timing: control (B separate shipped calls) vs batched ---- */
    printf("  %4s %14s %14s %10s\n", "B", "control ms/tok", "batched ms/tok", "speedup");

    const int Bs[] = { 1, 2, 4, 8, 16 };
    for (unsigned bi = 0; bi < sizeof(Bs)/sizeof(Bs[0]); ++bi) {
        const int B = Bs[bi];
        batch_fn bf = pick(B);
        double bestC = 1e30, bestB = 1e30;
        for (int r = 0; r < reps; ++r) {
            const int batched_first = (r & 1);   /* alternate order */
            for (int pass = 0; pass < 2; ++pass) {
                if ((pass == 0) == batched_first) {
                    const double t0 = now_s();
                    for (int t = 0; t < NT; ++t) {
                        W[t][(size_t)(r * 3 + t)] ^= 1;        /* anti-CSE */
                        for (int b = 0; b < B; ++b)
                            pq2_build_xp(x + (size_t)b * T[t].n,
                                         xp + (size_t)b * T[t].n, T[t].n);
                        bf(y, xp, W[t], T[t].n, T[t].d);
                        sink += y[(size_t)(r % B) * T[t].d + (r + t) % T[t].d];
                    }
                    const double dt = now_s() - t0;
                    if (dt < bestB) bestB = dt;
                } else {
                    const double t0 = now_s();
                    for (int t = 0; t < NT; ++t) {
                        W[t][(size_t)(r * 5 + t + 1)] ^= 1;
                        for (int b = 0; b < B; ++b) {
                            pq2_0_matvec_knc(yr, x + (size_t)b * T[t].n,
                                             W[t], T[t].n, T[t].d);
                            sink += yr[(r + b + t) % T[t].d];
                        }
                    }
                    const double dt = now_s() - t0;
                    if (dt < bestC) bestC = dt;
                }
            }
        }
        printf("  %4d %14.3f %14.3f %9.2fx\n",
               B, bestC / B * 1e3, bestB / B * 1e3, bestC / bestB);
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("control = shipped kernel x B calls (includes its own xp builds).\n");
    printf("batched = one decode pass for B tokens (xp builds included).\n");
    free(x); free(xp); free(y); free(yr);
    for (int t = 0; t < NT; ++t) free(W[t]);
    return 0;
}
