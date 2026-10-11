/* Is the matvec slow because of the KERNEL, or because of the per-CALL cost?
 *
 * Context (round-13 profile): mm_folded's matvec kernel is 39% of a token, the
 * largest single item. It streams 7.14 GB in 246 ms = 29.1 GB/s, only 14.8% of
 * the measured 196 GB/s ceiling -> NOT bandwidth-bound. It issues 40 vector ops
 * per 128 weights = 34.2 G ops/s = 0.45 ops/cycle/core against 0.99 measured for
 * a pure vpaddd stream at 4 threads/core -> ~2.2x issue capacity unclaimed.
 *
 * (K) kernel body stalls            -> reduce ops / restructure
 * (C) 401 calls/token, each entering and leaving a parallel region -> merge
 *
 * TRAPS BEING AVOIDED (both have produced fiction in this project before):
 *   1. looping the SAME call with unchanged arguments lets the compiler hoist
 *      or CSE it (that produced a bogus 101.7 G weights/s);
 *   2. allocating and freeing a tensor per call corrupted the heap in the first
 *      version of this file. Tensors are allocated ONCE up front instead, which
 *      is also what the model does (weights are mapped, not copied).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "pq2_0.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static volatile float sink;

/* calloc + explicit fill; aligned_alloc was suspected in the earlier crash and
 * is unnecessary here because pq2_0_matvec_knc only needs the raw bytes (the
 * 64-byte-aligned buffer it needs is wf, which is on its own stack). */
static uint8_t *mkweights(int n, int d) {
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t bytes = rb * (size_t)d;
    uint8_t *w = (uint8_t *)malloc(bytes);
    if (!w) { fprintf(stderr, "malloc %d x %d (%zu bytes) failed\n", n, d, bytes); exit(1); }
    uint32_t st = 12345u;
    for (size_t i = 0; i < bytes; ++i) { st = st * 1103515245u + 12345u; w[i] = (uint8_t)(st >> 16); }
    for (int r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + (size_t)r * rb + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;      /* fp16 scale = 1.0 */
        }
    return w;
}

/* the model's real folded shapes, in call order */
typedef struct { const char *nm; int n, d; } shp;
static const shp T[] = {
    { "ffn_gate",   5120, 17408 },
    { "ffn_up",     5120, 17408 },
    { "ffn_down",  17408,  5120 },
    { "attn_qkv",   5120, 10240 },
    { "attn_gate",  5120,  6144 },
};
#define NT ((int)(sizeof(T)/sizeof(T[0])))
#define MAXROW 17408

int main(int argc, char **argv) {
    const int nth  = argc > 1 ? atoi(argv[1]) : 244;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    omp_set_num_threads(nth);

    printf("=== matvec: kernel vs per-call cost ===\n");
    printf("threads=%d reps=%d procs=%d\n\n", nth, reps, omp_get_num_procs());

    /* Phase B packs ALL of phase A's weights into one n=17408 call, whose
     * row count (total_w / 17408 = 20178) exceeds MAXROW -- size y for that,
     * not for the widest phase-A output. (The first version overflowed y and
     * segfaulted on the card before printing anything.) */
    long total_w0 = 0;
    for (int t = 0; t < NT; ++t) total_w0 += (long)T[t].n * T[t].d;
    const size_t ymax = (size_t)(total_w0 / 17408) + 64;

    float *x = (float *)malloc((size_t)MAXROW * sizeof(float));
    float *y = (float *)malloc(ymax * sizeof(float));
    if (!x || !y) return 1;
    for (int i = 0; i < MAXROW; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;

    /* one weight buffer per distinct shape, allocated once */
    uint8_t *W[NT];
    for (int t = 0; t < NT; ++t) W[t] = mkweights(T[t].n, T[t].d);

    /* ---- A: separate calls, real shapes, like the model ---- */
    long total_w = 0;
    for (int t = 0; t < NT; ++t) total_w += (long)T[t].n * T[t].d;

    double bestA = 1e30;
    int ncalls = 0;
    for (int r = 0; r < reps; ++r) {
        const double a0 = now_s();
        for (int t = 0; t < NT; ++t) {
            W[t][(size_t)r] ^= 1;              /* defeat CSE/hoisting */
            pq2_0_matvec_knc(y, x, W[t], T[t].n, T[t].d);
            sink += y[(r * 7 + t) % (T[t].d - 1)];
        }
        const double a1 = now_s();
        if (a1 - a0 < bestA) bestA = a1 - a0;
        ncalls = NT;
    }
    printf("A: %d separate calls (the model's real shapes)\n", ncalls);
    printf("   best of %d: %.3f s   %.2f G weights   %.2f G w/s   %.3f ms/call\n\n",
           reps, bestA, total_w / 1e9, total_w / bestA / 1e9, bestA / ncalls * 1e3);

    /* ---- B: the same total weight count in ONE call ----
     * Rows are whatever makes n*d equal the phase-A total, using the widest n so
     * per-row work matches. This changes only the number of parallel regions. */
    {
        const int n = 17408;
        const int d = (int)(total_w / n);
        printf("B: ONE call, n=%d d=%d (%.2f G weights, exactly 1 parallel region)\n",
               n, d, (double)n * d / 1e9);
        uint8_t *w = mkweights(n, d);
        pq2_0_matvec_knc(y, x, w, n, d);        /* warm */
        double bestB = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double b0 = now_s();
            w[(size_t)r] ^= 1;
            pq2_0_matvec_knc(y, x, w, n, d);
            sink += y[r];
            const double b1 = now_s();
            if (b1 - b0 < bestB) bestB = b1 - b0;
        }
        printf("   best of %d: %.3f s   %.2f G w/s   %.3f ms\n\n",
               reps, bestB, (double)n * d / bestB / 1e9, bestB * 1e3);
        free(w);
    }

    printf("If A and B are close, the KERNEL dominates.\n");
    printf("If B is much faster, per-call / parallel-region cost dominates.\n");
    printf("sink=%.3f\n", (float)sink);
    for (int t = 0; t < NT; ++t) free(W[t]);
    free(x); free(y);
    return 0;
}
