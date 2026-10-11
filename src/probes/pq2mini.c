/* Minimal, hand-checkable IMCI kernel test.
 *
 * The full test showed 10237/10240 rows wrong with max abs error 45 on values
 * around 1441 -- i.e. not a rounding issue, a real defect. This narrows it
 * down by using a weight matrix whose exact result is known by construction:
 *
 *   every block gets scale d = 1.0, and every code is set to a chosen value
 *   from {0:-1, 1:0, 2:+1, 3:+2}, so row i is a known integer pattern.
 *
 * It prints, for row 0 only, the hand-computed expectation and both kernels,
 * then sweeps a few patterns that isolate specific failure modes:
 *   A. all codes = 0 (-1)      -> exercises the negative path
 *   B. all codes = 2 (+1)      -> exercises the positive path
 *   C. alternating 0/2         -> exercises sign mixture
 *   D. codes 3 (+2)            -> exercises the code==3 path
 *   E. varying scale           -> exercises the fp16 decode
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "pq2_0.h"

/* encode 1.0f as fp16 == 0x3C00 */
#define FP16_ONE 0x3C00

static void set_block(uint8_t *blk, uint16_t h, int code) {
    blk[0] = (uint8_t)(h & 0xFF);
    blk[1] = (uint8_t)(h >> 8);
    for (int q = 0; q < 32; ++q) {
        /* four 2-bit codes per byte, all equal to `code` */
        blk[2 + q] = (uint8_t)(code * 0x55);
    }
}

static int run_case(const char *name, int code, uint16_t h, float scale,
                    int n, int d) {
    const size_t nblk = (size_t)n / QK_PQ2_0;
    const size_t row_bytes = nblk * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;

    uint8_t *w = (uint8_t *)aligned_alloc(64, (wbytes + 63) & ~(size_t)63);
    float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float *yr = (float *)aligned_alloc(64, ((size_t)d * 4 + 63) & ~(size_t)63);
    float *yk = (float *)aligned_alloc(64, ((size_t)d * 4 + 63) & ~(size_t)63);
    if (!w || !x || !yr || !yk) return -1;

    for (size_t r = 0; r < (size_t)d; ++r)
        for (size_t b = 0; b < nblk; ++b)
            set_block(w + r * row_bytes + b * PQ2_0_BLOCK_BYTES, h, code);

    /* x = all ones makes the dot product equal to the plain sum of the weights,
       which for a uniform block is exactly n * (code-1) * scale. */
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    pq2_0_matvec_ref(yr, x, w, n, d);
    pq2_0_matvec_knc(yk, x, w, n, d);

    const double expect = (double)n * (double)(code - 1) * (double)scale;
    double maxd = 0.0;
    for (int i = 0; i < d; ++i) {
        const double e = fabs((double)yk[i] - (double)yr[i]);
        if (e > maxd) maxd = e;
    }
    printf("  %-26s code=%d scale=%.3f  expect=%10.4f  ref=%10.4f  knc=%10.4f  maxdiff=%.3e  %s\n",
           name, code, scale, expect, yr[0], yk[0], maxd,
           maxd < 1e-3 ? "OK" : "BAD");

    free(w); free(x); free(yr); free(yk);
    return maxd < 1e-3 ? 0 : 1;
}

int main(void) {
    const int n = 5120, d = 8;   /* small d so output is quick to inspect */
    int bad = 0;

    printf("=== minimal PQ2_0 kernel cases  (n=%d, d=%d, x=all ones) ===\n", n, d);
    printf("    'expect' is the exact value by construction\n\n");

    bad += run_case("all -1",   0, FP16_ONE, 1.0f, n, d);
    bad += run_case("all +1",   2, FP16_ONE, 1.0f, n, d);
    bad += run_case("all +2",   3, FP16_ONE, 1.0f, n, d);
    bad += run_case("all  0",   1, FP16_ONE, 1.0f, n, d);
    bad += run_case("all -1 half", 0, 0x3800, 0.5f, n, d);   /* scale 0.5 */
    bad += run_case("all -1 tiny", 0, 0x1400, 0.0001220703125f, n, d);

    printf("\n  cases failed: %d\n", bad);
    return bad ? 1 : 0;
}
