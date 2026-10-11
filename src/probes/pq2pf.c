/* Batch PREFILL: is there a win beyond what batch decode showed (1.11x)?
 *
 * Batch decode was measured at only 1.11x for B=16 (pq2batch.c) because the FMA
 * work grows linearly with B while only the decode is amortised, and the decode
 * is a minority of the ops once B is large.
 *
 * Prefill has a DIFFERENT reason to batch: weights. The current prefill path
 * calls bonsai_forward() once per prompt token, so a K-token prompt streams all
 * 7.14 GB of packed weights K times. If K tokens were processed against each
 * weight block while it is in registers, the weight traffic would be paid once
 * (7.14 GB) instead of K times.
 *
 * This measures that bandwidth-side win in isolation, separate from the decode
 * amortisation already shown to be small: for one weight row, how long does it
 * take to apply it to K activations, when the row is fetched once?
 *
 * Reference point: the measured streaming ceiling is 196 GB/s (rows-1360
 * pattern), and one token needs 7.14 GB = 36 ms at that rate. So if weight
 * traffic were the binding constraint, prefill of a K-token prompt should cost
 * ~36 ms + (compute), not K x as it does today.
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
    const int nth = argc > 1 ? atoi(argv[1]) : 244;
    const int K   = argc > 2 ? atoi(argv[2]) : 16;
    omp_set_num_threads(nth);

    /* one real tensor, so the numbers are in the model's own terms */
    const int n = 5120, d = 17408;
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t bytes = rb * (size_t)d;
    uint8_t *w = (uint8_t *)malloc(bytes + 256);
    float *x = (float *)aligned_alloc(64, (size_t)K * n * sizeof(float));
    float *y = (float *)aligned_alloc(64, (size_t)K * d * sizeof(float));
    if (!w || !x || !y) return 1;
    uint32_t st = 5u;
    for (size_t i = 0; i < bytes; ++i) { st = st*1103515245u+12345u; w[i] = (uint8_t)(st>>16); }
    for (int r = 0; r < d; ++r) { uint8_t *b = w + (size_t)r*rb; b[0]=0x00; b[1]=0x3C; }
    for (int i = 0; i < K * n; ++i) x[i] = (float)((i%7)-3)*0.25f;

    printf("=== prefill: weight traffic, K tokens, one %dx%d tensor ===\n", n, d);
    printf("threads=%d  K=%d  tensor=%.1f MB\n\n", nth, K, bytes/1e6);
    printf("  %6s %14s %14s %16s %14s\n",
           "K", "best (ms)", "GB streamed", "GB/s", "ms per token");

    double one_ms = 0;
    for (int k = 1; k <= K; k *= 2) {
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            w[(size_t)rep] ^= 1;
            const double t0 = now_s();
            /* Today's approach: one pass over the weights PER TOKEN. */
            for (int t = 0; t < k; ++t) {
                pq2_0_matvec_knc(y + (size_t)t * d, x + (size_t)t * n, w, n, d);
                sink += y[(size_t)t * d];
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        const double gb = bytes * (double)k / 1e9;      /* weights streamed k times */
        if (k == 1) one_ms = best * 1e3;
        printf("  %6d %14.2f %14.3f %16.2f %14.2f\n",
               k, best * 1e3, gb, gb / best, best * 1e3 / k);
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("\nThe 'ms per token' column is the tell: if it stays flat as K grows,\n");
    printf("weights are being reused from cache and batching buys nothing on the\n");
    printf("bandwidth side. If it stays equal to the K=1 value, the weights are\n");
    printf("re-streamed per token and a batched prefill could cut that by up to K.\n");
    free(w); free(x); free(y);
    return 0;
}
