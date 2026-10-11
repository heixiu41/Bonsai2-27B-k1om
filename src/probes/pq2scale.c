/* Why does the parallel model run each core at only 26-35% of the
 * single-thread kernel rate?
 *
 * Measured: model = 1.645 tok/s = 16.3 G weights/s over 61 cores
 *           = 268 M weights/s/core
 *           pq2ilp single-thread kernel body = 770-1047 M weights/s
 *         With 4 threads/core the model should reach ~3-4x the single-thread
 *         rate, not 0.3x. This measures the SHIPPED kernel's scaling curve to
 *         find where it goes wrong.
 *
 * Bug list from earlier attempts in this project, all avoided here:
 *   1. looping calls with unchanged arguments -> compiler hoists/CSEs them and
 *      the timing becomes fiction. Each repetition flips a weight byte.
 *   2. allocating/freeing a tensor per call corrupted the heap. Buffers are
 *      allocated once.
 *   3. discarding the result let the compiler delete the work. Everything is
 *      accumulated into a volatile sink that is printed.
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

static uint8_t *mkw(int n, int d) {
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t bytes = rb * (size_t)d;
    uint8_t *w = (uint8_t *)malloc(bytes);
    if (!w) { fprintf(stderr, "malloc %zu failed\n", bytes); exit(1); }
    uint32_t st = 12345u;
    for (size_t i = 0; i < bytes; ++i) { st = st*1103515245u + 12345u; w[i] = (uint8_t)(st>>16); }
    for (int r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + (size_t)r*rb + (size_t)b*PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;
        }
    return w;
}

int main(int argc, char **argv) {
    const int n = 5120, d = 17408;              /* the model's widest FFN tensor */
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const double weights = (double)n * d;

    float *x = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    float *y = (float *)aligned_alloc(64, (size_t)d * sizeof(float));
    uint8_t *w = mkw(n, d);
    if (!x || !y) return 1;
    for (int i = 0; i < n; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;

    printf("=== shipped kernel scaling: n=%d d=%d (%.2f G weights) ===\n", n, d, weights/1e9);
    printf("%8s %12s %14s %16s %12s\n",
           "threads", "best (ms)", "G weights/s", "per-core G w/s", "vs 1 thread");

    double base = 0;
    const int thr[] = { 1, 2, 4, 8, 16, 30, 61, 122, 244 };
    for (unsigned t = 0; t < sizeof(thr)/sizeof(thr[0]); ++t) {
        omp_set_num_threads(thr[t]);
        double best = 1e30;
        int nt = 0;
        #pragma omp parallel
        {
            #pragma omp master
            { nt = omp_get_num_threads(); }
        }
        for (int r = 0; r < reps; ++r) {
            w[(size_t)r] ^= 1;                   /* defeat hoisting */
            const double t0 = now_s();
            pq2_0_matvec_knc(y, x, w, n, d);
            const double t1 = now_s();
            sink += y[r];
            if (t1 - t0 < best) best = t1 - t0;
        }
        const double gws = weights / best / 1e9;
        if (thr[t] == 1) base = gws;
        const int cores = (nt + 3) / 4;          /* KNC: 4 threads per core */
        printf("%8d %12.3f %14.2f %16.2f %11.2fx\n",
               nt, best * 1e3, gws, gws / cores, gws / base);
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("\nReference points: pq2pipe measured a pure vpaddd stream at\n");
    printf("0.99 ops/cycle/core; the model's 244-thread rate corresponds to\n");
    printf("268 M weights/s/core = 0.084 ops/cycle/core at 40 ops/128 weights.\n");
    free(w); free(x); free(y);
    return 0;
}
