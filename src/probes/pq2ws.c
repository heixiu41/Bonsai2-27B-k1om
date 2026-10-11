/* Is the 4.3x gap between the isolated kernel and the model a WORKING-SET
 * effect (cache vs DRAM) or something else?
 *
 * Measured:
 *   pq2scale  : same kernel, same shape (5120x17408), 244 threads -> 111.65 G w/s
 *               weights buffer = 23.7 MB
 *   the model : matvec moves 6.80 GB/token in 0.263 s           ->  25.9 G w/s
 *               weights span 6.8 GB, a different tensor per call
 *
 * Both stream "26.87e9 weights per token", and at 25.9 G w/s that is 0.263 s.
 * Bandwidth used is only 25.9 GB/s of the 196 GB/s ceiling, so it is not raw
 * bandwidth. The obvious remaining difference is WHERE the weights come from:
 * pq2scale reuses one 23.7 MB tensor (L2-resident across the repeat loop),
 * while the model walks GB of cold DRAM.
 *
 * This measures the SAME kernel on the SAME shape while varying ONLY the size
 * of the weight region it cycles through -- so the only variable is cache
 * residency.
 *
 * Method notes (traps that produced fiction here before): the weight region is
 * allocated once, each repetition flips a byte so nothing can be hoisted, and
 * the result is accumulated into a printed volatile sink.
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
    const int n = 5120, d = 17408;
    const int nth = argc > 1 ? atoi(argv[1]) : 244;
    omp_set_num_threads(nth);

    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const double w_tensor = (double)n * d;            /* weights in one tensor */
    const size_t bytes1 = rb * (size_t)d;             /* bytes in one tensor */

    printf("=== same kernel, same shape, only the working set changes ===\n");
    printf("n=%d d=%d  one tensor = %.1f MB / %.3f G weights  threads=%d\n\n",
           n, d, bytes1 / 1e6, w_tensor / 1e9, nth);
    printf("  %8s %12s %14s %14s %12s\n",
           "copies", "region MB", "G weights/s", "per-core G w/s", "vs 1 copy");

    float *x = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    float *y = (float *)aligned_alloc(64, (size_t)d * sizeof(float));
    if (!x || !y) return 1;
    for (int i = 0; i < n; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;

    /* how many copies of the tensor to cycle through */
    const int copies[] = { 1, 4, 16, 64, 256 };
    double base = 0;
    for (unsigned ci = 0; ci < sizeof(copies)/sizeof(copies[0]); ++ci) {
        const int C = copies[ci];
        const size_t total = bytes1 * (size_t)C;
        if (total > 7.5e9) { printf("  (skipping %d copies: %.1f GB)\n", C, total/1e9); continue; }
        uint8_t *w = (uint8_t *)malloc(total + 256);
        if (!w) { printf("  (alloc %d copies failed)\n", C); continue; }
        uint32_t st = 12345u;
        for (size_t i = 0; i < total; ++i) { st = st*1103515245u+12345u; w[i] = (uint8_t)(st>>16); }
        for (int c = 0; c < C; ++c)
            for (int r = 0; r < d; ++r) {
                uint8_t *blk = w + (size_t)c*bytes1 + (size_t)r*rb;
                blk[0] = 0x00; blk[1] = 0x3C;
            }

        /* time C calls, one per copy, so each call touches a different region */
        const int reps = 3;
        double best = 1e30;
        for (int rep = 0; rep < reps; ++rep) {
            const double t0 = now_s();
            for (int c = 0; c < C; ++c) {
                w[(size_t)c * bytes1 + (size_t)(rep & 7)] ^= 1;   /* defeat hoisting */
                pq2_0_matvec_knc(y, x, w + (size_t)c * bytes1, n, d);
                sink += y[(c + rep) % d];
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        const double gws = w_tensor * C / best / 1e9;
        if (C == 1) base = gws;
        printf("  %8d %12.1f %14.2f %14.2f %11.2fx\n",
               C, total / 1e6, gws, gws / 61.0, gws / base);
        free(w);
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("\nIf G weights/s falls as the region grows, the gap is CACHE RESIDENCY\n");
    printf("(the model's 6.8 GB does not fit L2), not an instruction-count problem.\n");
    free(x); free(y);
    return 0;
}
