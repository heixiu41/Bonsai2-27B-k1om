/* probe_j.c -- confirm the default-disabled invariant: WITHOUT
 * BONSAI_IMCI_PQ2_0, pq2_0_matvec_knc must still be exactly the scalar
 * reference (so inference correctness is untouched by the fix).
 *
 * This is the same shape as pq2test.c's check, but it asserts the mapping
 * rather than just reporting agreement.
 *
 * Build for the card: icc -mmic -O2 -std=c11 probe_j.c -o probe_j -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "pq2_0.h"

static void *al(size_t bytes) {
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

static uint32_t g_state = 424242u;
static uint32_t rnd(void) {
    g_state = g_state * 1103515245u + 12345u;
    return (g_state >> 8);
}

int main(void) {
    const int n = 1280, d = 9;
    const size_t nblk = (size_t)n / QK_PQ2_0;
    const size_t row_bytes = nblk * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;
    uint8_t *w = (uint8_t *)al(wbytes);
    float *x = (float *)al((size_t)n * sizeof(float));
    float *yr = (float *)al((size_t)d * sizeof(float));
    float *yk = (float *)al((size_t)d * sizeof(float));

    for (size_t b = 0; b + PQ2_0_BLOCK_BYTES <= wbytes; b += PQ2_0_BLOCK_BYTES) {
        uint8_t *blk = w + b;
        const uint16_t h = (uint16_t)((5u + (rnd() % 20u)) << 10 | (rnd() & 0x3FFu));
        blk[0] = (uint8_t)(h & 0xFF);
        blk[1] = (uint8_t)(h >> 8);
        for (int q = 0; q < 32; ++q) blk[2 + q] = (uint8_t)(rnd() & 0xFF);
    }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(rnd() % 2000) - 1000) * 0.001f;

    pq2_0_matvec_ref(yr, x, w, n, d);
    pq2_0_matvec_knc(yk, x, w, n, d);

    double maxd = 0.0;
    for (int i = 0; i < d; ++i) maxd = fmax(maxd, fabs((double)yk[i] - (double)yr[i]));

    printf("=== probe_j: default-disabled invariant ===\n");
#ifdef BONSAI_IMCI_PQ2_0
    printf("  BONSAI_IMCI_PQ2_0 IS defined -> knc is the IMCI kernel\n");
#else
    printf("  BONSAI_IMCI_PQ2_0 is NOT defined -> knc must equal the reference\n");
    printf("  (on __MIC__ the macro below maps it to pq2_0_matvec_ref)\n");
#endif
    printf("  n=%d d=%d  max|knc-ref| = %.3e  %s\n", n, d, maxd,
           (maxd == 0.0) ? "IDENTICAL -> invariant holds" : "differs -> INVARIANT BROKEN");
    free(w); free(x); free(yr); free(yk);
    return (maxd == 0.0) ? 0 : 1;
}
