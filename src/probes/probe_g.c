/* probe_g.c -- irreducible per-accumulator facts, with volatile-forced storage.
 *
 * probe_e left a contradiction: it printed "sum0: 16 lanes all -1.0,
 * reduce=-16.0" yet "reduce(4 x all-(-1)) = -64 where -128 is correct".
 * Those cannot both be true of the same registers. This probe removes every
 * inference and prints, for the real 40-block loop:
 *
 *   for each accumulator: the reduce, the 16 stored lanes, the scalar sum,
 *                         and the low-128-bit (4 lane) sum
 *   for the combined vector: reduce vs scalar sum of 64 stored lanes
 *   and the sum of the four independent per-accumulator reduces
 *
 * Written with volatile stores so nothing can be constant-folded or reordered.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "pq2_0.h"

static void *al(size_t bytes) {
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

/* volatile so the compiler must really write the vector out */
static void vstore16(volatile float *dst, __m512 v) {
    float t[16] __attribute__((aligned(64)));
    _mm512_store_ps(t, v);
    for (int k = 0; k < 16; ++k) dst[k] = t[k];
}

static void report(const char *tag, __m512 v, double expect) {
    volatile float s[16];
    vstore16(s, v);
    double sc = 0.0, low4 = 0.0;
    for (int k = 0; k < 16; ++k) sc += (double)s[k];
    for (int k = 0; k < 4; ++k) low4 += (double)s[k];
    const double r = (double)_mm512_reduce_add_ps(v);
    printf("  %-8s lanes:", tag);
    for (int k = 0; k < 16; ++k) printf(" %5.1f", (double)s[k]);
    printf("\n           scalar=%9.1f  low4=%8.1f  4*low4=%9.1f  reduce=%9.1f"
           "   expect=%9.1f  %s\n",
           sc, low4, 4.0 * low4, r, expect,
           (fabs(r - expect) < 0.5) ? "reduce ok" : "reduce WRONG");
}

int main(void) {
    const int n = 5120;
    const int nblk = n / QK_PQ2_0;
    uint8_t *w = (uint8_t *)al((size_t)nblk * PQ2_0_BLOCK_BYTES);
    float   *x = (float *)al((size_t)n * sizeof(float));
    for (int b = 0; b < nblk; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;
        memset(blk + 2, 0x00, 32);
    }
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    float wf[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    int iters = 0;

    for (int j = 0; j < n; j += QK_PQ2_0) {
        ++iters;
        pq2_0_dequant_block(w + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), s3);
    }

    printf("=== probe_g: per-accumulator facts, n=%d, 40 blocks of -1 ===\n", n);
    printf("  iterations = %d (expect 40)\n", iters);
    printf("  expected per accumulator: 40 blocks x 16 lanes x -1 = -640\n");
    printf("  expected grand total    : -2560\n\n");

    report("sum0", s0, -640.0);
    report("sum1", s1, -640.0);
    report("sum2", s2, -640.0);
    report("sum3", s3, -640.0);

    /* the four independent reduces, summed in scalar */
    const double r0 = (double)_mm512_reduce_add_ps(s0);
    const double r1 = (double)_mm512_reduce_add_ps(s1);
    const double r2 = (double)_mm512_reduce_add_ps(s2);
    const double r3 = (double)_mm512_reduce_add_ps(s3);
    printf("\n  sum of 4 independent reduces = %.1f\n", r0 + r1 + r2 + r3);

    /* the combined vector the kernel actually reduces */
    __m512 t = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    report("combined", t, -2560.0);

    {
        volatile float s[16];
        vstore16(s, t);
        double sc = 0.0;
        for (int k = 0; k < 16; ++k) sc += (double)s[k];
        printf("\n  combined: scalar sum of stored lanes = %.1f\n", sc);
        printf("  combined: reduce                     = %.1f\n",
               (double)_mm512_reduce_add_ps(t));
    }

    /* all 64 lanes, summed scalar, from all four accumulators */
    {
        double sc = 0.0;
        volatile float s[16];
        vstore16(s, s0); for (int k = 0; k < 16; ++k) sc += (double)s[k];
        vstore16(s, s1); for (int k = 0; k < 16; ++k) sc += (double)s[k];
        vstore16(s, s2); for (int k = 0; k < 16; ++k) sc += (double)s[k];
        vstore16(s, s3); for (int k = 0; k < 16; ++k) sc += (double)s[k];
        printf("  all 64 stored lanes summed scalar    = %.1f  (correct -2560)\n", sc);
    }

    /* what the scalar reference says, for the record */
    {
        float y = 0.0f;
        pq2_0_matvec_ref(&y, x, w, n, 1);
        printf("  pq2_0_matvec_ref                     = %.1f\n", (double)y);
    }
    free(w); free(x);
    return 0;
}
