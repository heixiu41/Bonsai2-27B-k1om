/* probe_i.c -- validate the REAL pq2_0_matvec_knc from pq2_0.h, compiled with
 * BONSAI_IMCI_PQ2_0 defined, against pq2_0_matvec_ref over random data.
 *
 * Unlike pq2test.c (which maps pq2_0_matvec_knc to the reference when the macro
 * is undefined, so it silently compares the reference with itself), this file
 * forces the macro on, so both kernels are genuinely present and distinct.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define BONSAI_IMCI_PQ2_0 1
#include "pq2_0.h"

#ifdef BONSAI_IMCI_PQ2_0
#define KNC_IS_REAL 1
#else
#define KNC_IS_REAL 0
#endif

static void *al(size_t bytes) {
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

static uint32_t g_state = 13572468u;
static uint32_t rnd(void) {
    g_state = g_state * 1103515245u + 12345u;
    return (g_state >> 8);
}

static int one_case(int n, int d, const char *tag) {
    const size_t nblk = (size_t)n / QK_PQ2_0;
    const size_t row_bytes = nblk * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;
    uint8_t *w = (uint8_t *)al(wbytes);
    float *x = (float *)al((size_t)n * sizeof(float));
    float *yr = (float *)al((size_t)d * sizeof(float));
    float *yk = (float *)al((size_t)d * sizeof(float));
    if (!w || !x || !yr || !yk) return -1;

    for (size_t b = 0; b + PQ2_0_BLOCK_BYTES <= wbytes; b += PQ2_0_BLOCK_BYTES) {
        uint8_t *blk = w + b;
        const uint32_t sign = rnd() & 1u;
        const uint32_t exp  = 6u + (rnd() % 18u);
        const uint32_t mant = rnd() & 0x3FFu;
        const uint16_t h = (uint16_t)((sign << 15) | (exp << 10) | mant);
        blk[0] = (uint8_t)(h & 0xFF);
        blk[1] = (uint8_t)(h >> 8);
        for (int q = 0; q < 32; ++q) blk[2 + q] = (uint8_t)(rnd() & 0xFF);
    }
    for (int i = 0; i < n; ++i)
        x[i] = (float)((int)(rnd() % 4000) - 2000) * 0.0009f;

    pq2_0_matvec_ref(yr, x, w, n, d);
    pq2_0_matvec_knc(yk, x, w, n, d);

    double maxd = 0.0, maxr = 0.0;
    for (int i = 0; i < d; ++i) {
        maxd = fmax(maxd, fabs((double)yk[i] - (double)yr[i]));
        maxr = fmax(maxr, fabs((double)yr[i]));
    }
    const double rel = maxd / (maxr + 1e-30);
    const int ok = (rel < 1e-5);
    printf("  %-28s n=%5d d=%4d  |y|max=%9.3f  maxdiff=%.3e  rel=%.2e  %s\n",
           tag, n, d, maxr, maxd, rel, ok ? "AGREE" : "*** DISAGREE ***");
    free(w); free(x); free(yr); free(yk);
    return ok ? 0 : 1;
}

int main(void) {
    int bad = 0;
    printf("=== probe_i: real pq2_0_matvec_knc (header, BONSAI_IMCI_PQ2_0=1) ===\n");
    printf("    BONSAI_IMCI_PQ2_0 is defined: knc == %s\n\n",
           KNC_IS_REAL ? "the IMCI kernel (real)" : "the scalar reference (macro off)");
    if (!KNC_IS_REAL) { printf("  macro did not take effect\n"); return 2; }

    bad += one_case(128,   1, "single block");
    bad += one_case(256,   1, "2 blocks");
    bad += one_case(384,   3, "3 blocks");
    bad += one_case(1024,  8, "8 blocks");
    bad += one_case(1280, 17, "10 blocks, d=17");
    bad += one_case(2560,  5, "20 blocks");
    bad += one_case(5120,  8, "40 blocks");
    bad += one_case(5120, 64, "40 blocks, d=64");
    bad += one_case(17408, 6, "136 blocks (widest)");
    g_state = 55555u;
    printf("  -- second random pass --\n");
    bad += one_case(2688,  2, "21 blocks");
    bad += one_case(1024, 33, "8 blocks, d=33");
    bad += one_case(5120, 12, "40 blocks, d=12");

    printf("\n  failed cases: %d / 12\n", bad);
    return bad ? 1 : 0;
}
