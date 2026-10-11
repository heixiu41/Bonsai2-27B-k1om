/* Isolate the per-CALL parallel-region overhead, which is now the last
 * surviving explanation for the kernel-vs-model gap.
 *
 * ============================================================================
 * CORRECTION (after the probe results came in): the "4.3x gap" never existed.
 * The "model = 25.9 G weights/s" figure below was a UNIT ERROR: 6.80 GB per
 * token / 0.263 s = 25.9 **GB/s of packed bytes**, and PQ2_0 packs 128 weights
 * into 34 bytes, so the model's true rate was 25.9 * 128/34 = **97.5 G
 * weights/s** -- within 15% of the isolated 111.65 G w/s, and essentially
 * identical to the isolated kernel measured on the model's REAL shape mix
 * (pq2call phase A: 99.06 G w/s). The "gap" was exactly the bytes<->weights
 * conversion factor (3.76x). This probe's own result (region overhead =
 * 401 x 34 us = 13.6 ms/token) remains valid.
 * ============================================================================
 *
 * Established so far:
 *   pq2scale : shipped kernel, 5120x17408, 244 threads -> 111.65 G weights/s
 *   pq2ws    : same kernel, region grown 23.7 MB -> 6 GB, throughput falls only
 *              3.6% (111.8 -> 107.8 G w/s)  => NOT a cache/working-set effect
 *   model    : moves 6.80 GB of matvec weights per token in 0.263 s
 *              = 25.9 G weights/s over 401 calls
 *   => 4.3x gap, and the only structural difference left is that the model
 *      enters and leaves a `#pragma omp parallel for` 401 times per token
 *      while the benchmarks do it a handful of times.
 *
 * THE EXPERIMENT. Same kernel, same total weight count, same thread count.
 * Vary ONLY the number of parallel regions the work is split across, by
 * splitting one tensor into P row-blocks and calling the kernel P times.
 * If wall time rises with P, the difference is per-region cost and it is
 * measurable in isolation.
 *
 * Traps avoided, all of which bit earlier attempts here:
 *   - each call gets a different pointer, so nothing can be hoisted or CSE'd;
 *   - buffers allocated once, never per call (an earlier version corrupted the
 *     heap doing that);
 *   - results accumulated into a printed volatile sink, so the work survives
 *     optimisation;
 *   - the timing uses clock_gettime, not omp_get_wtime (unusable on this target).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "pq2_0.h"

static volatile float sink;
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const int n = 5120, d = 17408;               /* widest FFN shape */
    const int nth = argc > 1 ? atoi(argv[1]) : 244;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    omp_set_num_threads(nth);

    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t total_bytes = rb * (size_t)d;
    uint8_t *w = (uint8_t *)malloc(total_bytes + 256);
    float *x = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    float *y = (float *)aligned_alloc(64, (size_t)d * sizeof(float));
    if (!w || !x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }
    uint32_t st = 12345u;
    for (size_t i = 0; i < total_bytes; ++i) { st = st*1103515245u+12345u; w[i] = (uint8_t)(st>>16); }
    for (int r = 0; r < d; ++r) { uint8_t *b = w + (size_t)r*rb; b[0]=0x00; b[1]=0x3C; }
    for (int i = 0; i < n; ++i) x[i] = (float)((i%7)-3)*0.25f;

    printf("=== per-call parallel-region cost, isolated ===\n");
    printf("shape %dx%d (%.3f G weights), threads=%d, reps=%d\n\n",
           n, d, (double)n*d/1e9, nth, reps);
    printf("  %8s %14s %14s %14s %16s\n",
           "P blocks", "rows/block", "best (ms)", "G weights/s", "us per region");

    const int P[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    double base_ms = 0;
    for (unsigned pi = 0; pi < sizeof(P)/sizeof(P[0]); ++pi) {
        const int p = P[pi];
        if (p > d) break;
        const int rows = d / p;                  /* rows per call */
        if (rows < 1) break;

        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            for (int b = 0; b < p; ++b) {
                const int off = b * rows;
                w[(size_t)off * rb + (size_t)(r & 7)] ^= 1;   /* vary per call */
                pq2_0_matvec_knc(y + off, x, w + (size_t)off * rb, n, rows);
                sink += y[off];
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        const double ms = best * 1e3;
        const double gws = (double)n * d / best / 1e9;
        if (p == 1) base_ms = ms;
        /* per-region cost = (time at P regions - time at 1 region) / (P-1) */
        const double per_region_us = p > 1 ? (ms - base_ms) / (p - 1) * 1e3 : 0.0;
        printf("  %8d %14d %14.3f %14.2f %16.1f\n",
               p, rows, ms, gws, per_region_us);
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("\nIf 'us per region' is stable and non-trivial, THAT is the cost the\n");
    printf("model pays 401 times a token. 401 x it = ms/token of pure overhead.\n");
    free(w); free(x); free(y);
    return 0;
}
