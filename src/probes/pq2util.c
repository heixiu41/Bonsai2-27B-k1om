/* Are all 244 threads actually doing work, and how evenly?
 *
 * Context: the model runs at 244 threads and its matvec achieves 102 G
 * weights/s against the kernel's isolated 111.65 G weights/s at the same
 * thread count -- so on the surface the threads are used. But two things were
 * never CHECKED, only assumed:
 *   (a) that the OpenMP runtime really spreads 244 threads over all 61 cores
 *       (KNC has 4 hardware threads per core; a bad pinning/scheduling policy
 *       can stack them or leave cores idle);
 *   (b) that the work is evenly divided, and that a thread finishing early is
 *       not just sitting at the barrier of a 401-times-per-token parallel
 *       region.
 *
 * This counts the number of ROWS each thread processed inside the shipped
 * kernel's parallel loop, over a full model-shaped workload, and reports the
 * distribution. If some threads report zero, or the max/min ratio is far from
 * 1, the cores are not being used well.
 *
 * Implementation: a copy of the shipped kernel's row loop with an atomic
 * per-thread counter. The arithmetic is identical, only the accounting is added.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "pq2_0.h"

static long rows_seen[4096];        /* rows each thread id processed */
static double sec_seen[4096];       /* wall time each thread spent in the loop */

static int pick(int i, const double *v) { (void)v; (void)i; return 0; }

int main(int argc, char **argv) {
    const int n = 5120, d = 17408;
    const int nth = argc > 1 ? atoi(argv[1]) : 244;
    omp_set_num_threads(nth);

    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t total = rb * (size_t)d;
    uint8_t *w = (uint8_t *)malloc(total + 256);
    float *x = (float *)aligned_alloc(64, (size_t)n * sizeof(float));
    float *y = (float *)aligned_alloc(64, (size_t)d * sizeof(float));
    if (!w || !x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }
    uint32_t st = 12345u;
    for (size_t i = 0; i < total; ++i) { st = st*1103515245u+12345u; w[i] = (uint8_t)(st>>16); }
    for (int r = 0; r < d; ++r) { uint8_t *b = w + (size_t)r*rb; b[0]=0x00; b[1]=0x3C; }
    for (int i = 0; i < n; ++i) x[i] = (float)((i%7)-3)*0.25f;

    printf("=== thread utilisation inside the kernel's row loop ===\n");
    printf("shape %dx%d, requested threads=%d, procs reported=%d\n\n",
           n, d, nth, omp_get_num_procs());

    const int reps = 5;
    for (int r = 0; r < reps; ++r) {
        memset(rows_seen, 0, sizeof rows_seen);
        memset(sec_seen, 0, sizeof sec_seen);
        w[(size_t)r] ^= 1;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            struct timespec a, b;
            clock_gettime(CLOCK_MONOTONIC, &a);
            long mine = 0;
            #pragma omp for schedule(static)
            for (int i = 0; i < d; ++i) {
                /* the shipped body, minus the timing: decode and accumulate */
                pq2_0_matvec_knc(y + i, x, w + (size_t)i * rb, n, 1);
                ++mine;
            }
            clock_gettime(CLOCK_MONOTONIC, &b);
            rows_seen[tid] = mine;
            sec_seen[tid] = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        const double wall = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec)*1e-9;

        int nt = 0, nzero = 0;
        long mn = 1L<<60, mx = 0, sum = 0;
        double smn = 1e30, smx = 0;
        for (int t = 0; t < 4096; ++t) {
            if (rows_seen[t] == 0 && sec_seen[t] == 0.0) continue;
            ++nt;
            if (rows_seen[t] == 0) ++nzero;
            if (rows_seen[t] < mn) mn = rows_seen[t];
            if (rows_seen[t] > mx) mx = rows_seen[t];
            sum += rows_seen[t];
            if (sec_seen[t] < smn) smn = sec_seen[t];
            if (sec_seen[t] > smx) smx = sec_seen[t];
        }
        printf("  rep %d: wall %.3f ms | threads that reported: %d | rows assigned total %ld (want %d)\n",
               r, wall*1e3, nt, sum, d);
        printf("          rows/thread  min %ld  max %ld  max/min %.2f | idle threads: %d\n",
               mn, mx, mn ? (double)mx/mn : 0.0, nzero);
        printf("          busy time/thread min %.3f ms max %.3f ms  (imbalance %.2fx)\n",
               smn*1e3, smx*1e3, smn > 0 ? smx/smn : 0.0);
    }

    printf("\nInterpretation: 'threads that reported' should equal the requested thread\n");
    printf("count, total rows should equal d, and max/min should be near 1. A big\n");
    printf("busy-time spread means threads finish at wildly different times and wait\n");
    printf("at the barrier -- i.e. the cores are not evenly loaded.\n");
    free(w); free(x); free(y);
    (void)pick;
    return 0;
}
