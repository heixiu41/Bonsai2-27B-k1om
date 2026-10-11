/* Why is the decoder at 0.18 ops/cycle/core when 0.99 is achievable?
 *
 * Measured context:
 *   per token   : 26.87e9 weights / 128 * N_OPS vector ops
 *   measured    : 1.645 tok/s at 244 threads, 61 cores, 1.238 GHz
 *   => 13.8 G vector ops/s = 0.18 ops/cycle/core
 *   pq2pipe.c measured a pure vpaddd stream at 0.99 ops/cycle/core.
 *   So the decoder uses ~18% of the issue capacity the hardware demonstrably has.
 *
 * Two candidate causes, and they need different fixes:
 *   (D) DEPENDENCY LATENCY. The per-group chain is
 *         LOAD16 -> vpermd -> vpsrlvd -> vpermd(LUT) -> vmulps -> FMA
 *       which is ~5 deep, and vpermd is microcoded with long latency. With one
 *       chain in flight a thread cannot issue every cycle.
 *   (E) it is not latency but something else in the real kernel's shape
 *       (addressing, loads, the OpenMP row loop).
 *
 * This file measures the decoder's INNER BODY in isolation, at the same
 * instruction mix, with a controllable number of INDEPENDENT BLOCKS in flight.
 * If throughput rises with blocks-in-flight up to the core count, the cause is
 * (D) and the fix is more independent work per thread. If it is flat, the body
 * is already saturated at its instruction mix and only count reduction helps.
 *
 * TRAPS AVOIDED: every variant reads DIFFERENT weight bytes (no hoisting), the
 * result is folded into a printed sink, and blocks-in-flight is the only
 * variable that changes between runs.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#include "pq2_0.h"

static volatile float sink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- the decoder inner body, exactly as the shipped kernel has it ----
 * One 128-weight block -> 8 float vectors, each consumed by an FMA. */
static inline __m512 block_decode(const uint8_t *blk, const float *x, int idx) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const __m512i shf = PQ2_IMCI_SHIFT();
    const __m512  vd  = _mm512_set1_ps(d);
    const __m512i c0 = PQ2_IMCI_LOAD16(qs);
    const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
    __m512 acc = _mm512_setzero_ps();
    #define G(m, c, off) do {                                        \
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c);\
        const __m512i v = _mm512_srlv_epi32(b, shf);                 \
        const __m512  g = _mm512_mul_ps(                              \
            _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);\
        acc = _mm512_fmadd_ps(g, _mm512_load_ps(x + (off)), acc);     \
    } while (0)
    G(0, c0,   0); G(1, c0,  16); G(2, c0,  32); G(3, c0,  48);
    G(0, c1,  64); G(1, c1,  80); G(2, c1,  96); G(3, c1, 112);
    #undef G
    (void)idx;
    return acc;
}

/* Run `k` independent blocks per outer iteration: k = ILP. */
static double bench(int k, int nblocks, int reps) {
    const size_t bytes = (size_t)nblocks * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)malloc(bytes + 256);
    float   *x = (float *)aligned_alloc(64, 1024 * sizeof(float));
    if (!w || !x) { fprintf(stderr, "alloc failed\n"); exit(1); }
    uint32_t st = 7u;
    for (size_t i = 0; i < bytes; ++i) { st = st * 1103515245u + 12345u; w[i] = (uint8_t)(st >> 16); }
    for (int b = 0; b < nblocks; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;                  /* fp16 1.0 */
    }
    for (int i = 0; i < 1024; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;

    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const int per = nblocks / k;
        const double t0 = now_s();
        for (int i = 0; i + k <= per * k; i += k) {
            __m512 a = _mm512_setzero_ps();
            for (int j = 0; j < k; ++j) {
                const uint8_t *blk = w + (size_t)(i + j) * PQ2_0_BLOCK_BYTES;
                /* rotate which 16-float window each block uses: makes the loads
                 * distinct and cannot be hoisted */
                a = _mm512_add_ps(a, block_decode(blk, x + 16 * ((i + j) & 3), i + j));
            }
            sink += _mm512_reduce_add_ps(a);
        }
        const double t1 = now_s();
        if (t1 - t0 < best) best = t1 - t0;
    }
    free(w); free(x);
    /* 40 vector ops per 128 weights is the count from the body above:
     * 4 LOAD16 (8 ops) + 8 vpermd + 8 vpsrlvd + 8 vmulps + 8 FMA = 40 */
    const double weights = (double)(nblocks / k) * k * 128;
    return weights / best;                  /* weights per second */
}

int main(int argc, char **argv) {
    const int nth   = argc > 1 ? atoi(argv[1]) : 1;
    const int nblk  = argc > 2 ? atoi(argv[2]) : 16384;
    const int reps  = argc > 3 ? atoi(argv[3]) : 3;
    omp_set_num_threads(nth);

    printf("=== decoder inner body: throughput vs blocks-in-flight (ILP) ===\n");
    printf("threads=%d blocks=%d reps=%d freq=1.238GHz\n\n", nth, nblk, reps);
    printf("  %-8s %14s %12s %18s\n", "k (ILP)", "M weights/s", "ops/cycle/core", "ops/s (per core)");

    for (int k = 1; k <= 8; k *= 2) {
        double best = 0;
        #pragma omp parallel num_threads(nth) reduction(max:best)
        {
            const double r = bench(k, nblk, reps);
            if (r > best) best = r;
        }
        /* ops per 128 weights = 40; per core = best/61 weights/s */
        const double ops_per_core = best / 61.0 * (40.0 / 128.0);
        printf("  %-8d %14.1f %12.3f %18.2e\n",
               k, best / 1e6, ops_per_core / 1.238e9, ops_per_core);
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("\nIf ops/cycle/core RISES with k, the decoder is dependency-latency\n");
    printf("bound and the fix is more independent work per thread (batch rows or\n");
    printf("tokens). If it is FLAT around 0.99, the body is issue-saturated and\n");
    printf("only reducing the op count helps.\n");
    return 0;
}
