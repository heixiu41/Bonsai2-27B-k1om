/* On-card differential test: IMCI PQ2_0 kernel vs the scalar reference.
 *
 * Both are compiled into this binary. It fills a synthetic PQ2_0 matrix and a
 * matching input vector, runs both paths over the SAME bytes, and prints the
 * first differing element plus the norms. If the two disagree here, the bug is
 * in the IMCI kernel; if they agree, the bug is elsewhere in the pipeline.
 *
 * Usage: pq2test [n] [d]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "pq2_0.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* simple reproducible PRNG so both runs see identical bytes */
static uint32_t g_state = 12345u;
static uint32_t rnd(void) {
    g_state = g_state * 1103515245u + 12345u;
    return (g_state >> 8) & 0xFFFFFF;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 5120;   /* input dim  */
    const int d = argc > 2 ? atoi(argv[2]) : 10240;  /* output dim */
    if (n % QK_PQ2_0) { fprintf(stderr, "n must be a multiple of %d\n", QK_PQ2_0); return 2; }

    const size_t nblk_per_row = (size_t)n / QK_PQ2_0;
    const size_t row_bytes = nblk_per_row * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;

    printf("=== PQ2_0 IMCI vs scalar reference ===\n");
    printf("  n=%d d=%d   weight bytes=%zu (%.1f MB)\n", n, d, wbytes, wbytes / 1e6);

    /* 64-byte aligned weight + input + outputs, so alignment is not a variable */
    uint8_t *w = (uint8_t *)aligned_alloc(64, (wbytes + 63) & ~(size_t)63);
    float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float *y_ref = (float *)aligned_alloc(64, ((size_t)d * 4 + 63) & ~(size_t)63);
    float *y_knc = (float *)aligned_alloc(64, ((size_t)d * 4 + 63) & ~(size_t)63);
    if (!w || !x || !y_ref || !y_knc) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* fill weights with valid PQ2_0 blocks: fp16 scale + 2-bit codes */
    for (size_t r = 0; r < (size_t)d; ++r) {
        for (size_t b = 0; b < nblk_per_row; ++b) {
            uint8_t *blk = w + r * row_bytes + b * PQ2_0_BLOCK_BYTES;
            /* a small positive fp16 scale: pick a few representative values */
            const float sc = 0.001f + (float)(rnd() % 1000) * 0.0005f;
            /* encode sc as fp16 approximately */
            const uint32_t fb = *(const uint32_t *)&sc;
            const uint16_t h = (uint16_t)(((fb >> 16) & 0x8000) |
                                          ((((fb >> 23) & 0xFF) - 127 + 15) << 10) |
                                          ((fb >> 13) & 0x3FF));
            blk[0] = (uint8_t)(h & 0xFF);
            blk[1] = (uint8_t)(h >> 8);
            for (int q = 0; q < 32; ++q) blk[2 + q] = (uint8_t)(rnd() & 0xFF);
        }
    }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(rnd() % 2000) - 1000) * 0.001f;

    double t0 = now_s();
    pq2_0_matvec_ref(y_ref, x, w, n, d);
    double t1 = now_s();
    pq2_0_matvec_knc(y_knc, x, w, n, d);
    double t2 = now_s();

    printf("  ref  %.3f s   knc %.3f s\n", t1 - t0, t2 - t1);

    int nbad = 0, first = -1;
    double maxabs = 0.0;
    for (int i = 0; i < d; ++i) {
        const double diff = fabs((double)y_knc[i] - (double)y_ref[i]);
        if (diff > maxabs) maxabs = diff;
        if (diff > 1e-3 * (fabs((double)y_ref[i]) + 1e-6)) {
            if (first < 0) first = i;
            ++nbad;
        }
    }
    printf("\n  mismatching rows : %d / %d\n", nbad, d);
    printf("  max abs diff     : %.6e\n", maxabs);
    if (first >= 0) {
        printf("  first mismatch at row %d:\n", first);
        printf("      ref = %.6f\n", y_ref[first]);
        printf("      knc = %.6f\n", y_knc[first]);
    }

    /* norm comparison catches a systematic scale error that per-element
       tolerances might hide */
    double sref = 0.0, sknc = 0.0;
    for (int i = 0; i < d; ++i) { sref += (double)y_ref[i]*y_ref[i]; sknc += (double)y_knc[i]*y_knc[i]; }
    printf("  ||y_ref|| = %.4f   ||y_knc|| = %.4f   ratio = %.6f\n",
           sqrt(sref), sqrt(sknc), sqrt(sknc) / (sqrt(sref) + 1e-30));

    printf("\n  VERDICT: %s\n", nbad == 0 ? "kernels AGREE -> bug is elsewhere"
                                          : "kernels DISAGREE -> bug is in the IMCI kernel");
    free(w); free(x); free(y_ref); free(y_knc);
    return nbad == 0 ? 0 : 1;
}
