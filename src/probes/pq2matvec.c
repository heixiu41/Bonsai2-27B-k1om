/* pq2matvec.c -- full PQ2_0 matvec on the card, single thread.
 *
 * Build it twice:
 *   icc -mmic -O2 -std=c11 src/bonsai/pq2matvec.c -o pq2matvec_ref  -lm
 *   icc -mmic -O2 -std=c11 -DBONSAI_IMCI_PQ2_0 src/bonsai/pq2matvec.c \
 *       -o pq2matvec_imci -lm
 * The first exercises the DEFAULT (scalar) kernel, the second the IMCI kernel
 * now that it calls pq2_0_dequant_block_imci.  Both kernels are always
 * reachable in the same binary (pq2_0_matvec_ref is compiled on every target),
 * so the IMCI build also checks itself against the scalar reference.
 *
 * Without -openmp the `#pragma omp parallel for` is ignored, which is exactly
 * the single-thread measurement we want.
 *
 * Uses the real 10240x5120 model-matrix shape, so the packed weights (13.3 MB)
 * stream from main memory rather than fitting in the 512 KB L2.
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

static uint32_t rng = 77u;
static uint32_t xr(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int n = 10240, d = 5120;
    if (argc > 2) { n = atoi(argv[1]); d = atoi(argv[2]); }

    const int    nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;
    uint8_t *w;
    float *x, *y_ref, *y_knc;
    int reps = 3;
    double t0, t1, t_ref, t_knc;
    long mism = 0;
    double maxdiff = 0.0, maxrel = 0.0, ymax = 0.0;

    /* +512 slack: the IMCI decoder's lo/hi load pair reaches up to ~96 bytes
     * past the last block payload. */
    w     = (uint8_t *)aligned_alloc(64, ((wbytes + 512) + 63) & ~(size_t)63);
    x     = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    y_ref = (float *)aligned_alloc(64, ((size_t)d * 4 + 63) & ~(size_t)63);
    y_knc = (float *)aligned_alloc(64, ((size_t)d * 4 + 63) & ~(size_t)63);
    if (!w || !x || !y_ref || !y_knc) { printf("alloc failed\n"); return 1; }

    for (size_t b = 0; b < (size_t)d * nblk; ++b) {
        uint8_t *blk = w + b * PQ2_0_BLOCK_BYTES;
        /* finite fp16 scale: exp in 0..30 */
        uint16_t h = (uint16_t)(((xr() & 1) << 15) | ((1 + xr() % 30) << 10) | (xr() & 0x3FF));
        blk[0] = (uint8_t)(h & 0xFF);
        blk[1] = (uint8_t)(h >> 8);
        for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(xr() & 0xFF);
    }
    for (int j = 0; j < n; ++j) x[j] = (float)((int)(xr() % 2001) - 1000) / 1000.0f;

    printf("PQ2_0 matvec on KNC: n=%d d=%d  (%d weights, %.2f MB packed), "
           "%d reps, single thread (no -openmp)\n",
           n, d, n * d, (double)wbytes / 1048576.0, reps);
#ifdef BONSAI_IMCI_PQ2_0
    printf("kernel under test: pq2_0_matvec_knc  == IMCI path (pq2_0_dequant_block_imci)\n");
#else
    printf("kernel under test: pq2_0_matvec_knc  == DEFAULT scalar path (pq2_0_matvec_ref)\n");
#endif

    /* scalar reference, always available */
    t0 = now_s();
    for (int r = 0; r < reps; ++r) pq2_0_matvec_ref(y_ref, x, w, n, d);
    t_ref = (now_s() - t0) / reps;

    /* the kernel selected at compile time */
    t1 = now_s();
    for (int r = 0; r < reps; ++r) pq2_0_matvec_knc(y_knc, x, w, n, d);
    t_knc = (now_s() - t1) / reps;

    for (int i = 0; i < d; ++i) {
        const double e = fabs((double)y_ref[i] - (double)y_knc[i]);
        const double rel = e / (fabs((double)y_ref[i]) + 1e-30);
        if (e > maxdiff) maxdiff = e;
        if (rel > maxrel) maxrel = rel;
        if (rel > 1e-3) ++mism;
    }
    for (int i = 0; i < d; ++i) {
        if (fabs((double)y_ref[i]) > ymax) ymax = fabs((double)y_ref[i]);
    }

    printf("\n  pq2_0_matvec_ref  : %8.4f s  %8.2f M weights/s  (scalar decode + scalar MAC)\n",
           t_ref, (double)n * d / t_ref / 1e6);
    printf("  pq2_0_matvec_knc  : %8.4f s  %8.2f M weights/s\n",
           t_knc, (double)n * d / t_knc / 1e6);
    printf("  matvec speedup    : %.2fx\n", t_ref / t_knc);
    printf("\n  |y|max = %.6g\n", ymax);
    printf("  knc - ref : max abs diff = %.6g, max REL diff = %.3g, "
           "rows with rel > 1e-3 = %ld / %d\n", maxdiff, maxrel, mism, d);
    printf("  %s\n", (mism == 0 && maxrel < 1e-3)
           ? "IMCI matvec matches the scalar reference (float summation-order noise only)"
           : "IMCI matvec DISAGREES with the scalar reference beyond float noise");

    free(w); free(x); free(y_ref); free(y_knc);
    return 0;
}
