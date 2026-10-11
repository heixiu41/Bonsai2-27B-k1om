/* probe_d.c -- decisive decomposition of the half-value defect.
 *
 * Contradiction to resolve: probe_a's per-lane dump shows all 16 lanes of all
 * four accumulators at exactly -40 after 40 blocks (so each accumulator's sum
 * is -640 and the grand total -5120), yet the same program returns -2560.
 * Either the dump is lying about what the accumulators hold, or the final
 * combination is not the sum of the dumps.
 *
 * This probe instruments PER BLOCK so the exact place where the running total
 * stops matching ~128 * (blocks so far) is visible, and cross-checks
 * _mm512_reduce_add_ps against a store+scalar sum of the SAME register value.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "pq2_0.h"

#define FP16_ONE 0x3C00

static void *al(size_t bytes) {
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

/* all weights -1.0 */
static void fill_m1(uint8_t *w, int nblk) {
    for (int b = 0; b < nblk; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;
        memset(blk + 2, 0x00, 32);
    }
}

static double sym(const char *tag, __m512 v) {
    float s[16] __attribute__((aligned(64)));
    _mm512_store_ps(s, v);
    double acc = 0.0;
    for (int k = 0; k < 16; ++k) acc += (double)s[k];
    const double r = (double)_mm512_reduce_add_ps(v);
    printf("      %-22s reduce=%10.1f  store+scalar=%10.1f  %s\n",
           tag, r, acc, (fabs(r - acc) < 0.5) ? "same" : "*** DIFFER ***");
    return r;
}

/* ---- per-block running total (x = all ones) ---- */
static void per_block(int n, int trace_every) {
    const int nblk = n / QK_PQ2_0;
    uint8_t *w = (uint8_t *)al((size_t)nblk * PQ2_0_BLOCK_BYTES);
    float   *x = (float *)al((size_t)n * sizeof(float));
    fill_m1(w, nblk);
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    float wf[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    double prev = 0.0;
    int first_divergence = -1;

    printf("\n=== per-block running total, n=%d (%d blocks), all w=-1, x=1 ===\n",
           n, nblk);
    printf("    correct running total after b blocks = %d * (b+1)\n", -n / nblk);
    printf("    b    j      reduce(total)   store(total)   correct     delta\n");

    for (int b = 0; b < nblk; ++b) {
        const int j = b * QK_PQ2_0;
        pq2_0_dequant_block(w + (size_t)b * PQ2_0_BLOCK_BYTES, wf);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), s3);

        __m512 t = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
        const double r = (double)_mm512_reduce_add_ps(t);
        const double correct = (double)(-n / nblk) * (double)(b + 1);

        int show = (b < 3) || (b >= nblk - 3) || (b % trace_every == 0);
        if (show)
            printf("  %3d %5d  %12.1f  %12.1f  %10.1f  %8.1f\n",
                   b, j, r, (double)0, correct, r - correct);
        if (b > 0 && first_divergence < 0 && fabs(r - correct) > 0.5)
            first_divergence = b;
        prev = r;
        (void)prev;
    }

    printf("\n  per-accumulator check after the loop (all four must agree):\n");
    sym("sum0", s0); sym("sum1", s1); sym("sum2", s2); sym("sum3", s3);
    printf("\n  first block index where the running total diverges: %d\n",
           first_divergence);
    printf("  (0 => the very first block is already wrong; -1 => never diverges)\n");
    free(w); free(x);
}

/* ---- per-block weights: block b sums to -128*(b+1) ----
 * If the loop only ever sees every other block (stride 256) or only half the
 * lanes of x, the total identifies which. */
static void distinct_blocks(void) {
    const int n = 5120, nblk = n / QK_PQ2_0;
    uint8_t *w = (uint8_t *)al((size_t)nblk * PQ2_0_BLOCK_BYTES);
    float   *x = (float *)al((size_t)n * sizeof(float));
    fill_m1(w, nblk);                       /* all weights -1 */
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    float wf[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int b = 0; b < nblk; ++b) {
        const int j = b * QK_PQ2_0;
        /* rewrite x for this block: constant value (b+1) inside the block */
        for (int k = 0; k < QK_PQ2_0; ++k) x[j + k] = (float)(b + 1);
        pq2_0_dequant_block(w + (size_t)b * PQ2_0_BLOCK_BYTES, wf);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), s3);
    }
    const double got = (double)_mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
    double all = 0.0, odd = 0.0, even = 0.0, first20 = 0.0, last20 = 0.0;
    for (int b = 0; b < nblk; ++b) {
        const double c = -128.0 * (double)(b + 1);
        all += c;
        if (b & 1) odd += c; else even += c;
        if (b < 20) first20 += c; else last20 += c;
    }
    printf("\n=== distinct per-block weights (block b sum = -128*(b+1)) ===\n");
    printf("  all 40 blocks        : %.1f\n", all);
    printf("  only odd blocks      : %.1f\n", odd);
    printf("  only even blocks     : %.1f\n", even);
    printf("  first 20 blocks      : %.1f\n", first20);
    printf("  last  20 blocks      : %.1f\n", last20);
    printf("  half of all (every other x lane): %.1f\n", all / 2.0);
    printf("  KERNEL GOT           : %.1f\n", got);
    printf("  -> %s\n", (fabs(got - all) < 0.5) ? "matches ALL blocks" :
           (fabs(got - odd) < 0.5)     ? "matches ODD blocks only" :
           (fabs(got - even) < 0.5)    ? "matches EVEN blocks only" :
           (fabs(got - all/2) < 0.5)   ? "matches HALF of every block" :
                                          "matches nothing simple");
    free(w); free(x);
}

/* ---- is _mm512_store_ps writing all 16 lanes? ---- */
static void store_check(void) {
    float v[16] __attribute__((aligned(64)));
    for (int k = 0; k < 16; ++k) v[k] = 1000.0f + (float)k;
    __m512 vec = _mm512_load_ps(v);
    float out[16] __attribute__((aligned(64)));
    for (int k = 0; k < 16; ++k) out[k] = -999.0f;
    _mm512_store_ps(out, vec);
    printf("\n=== _mm512_store_ps lane coverage ===\n  in : ");
    for (int k = 0; k < 16; ++k) printf("%.0f ", (double)v[k]);
    printf("\n  out: ");
    for (int k = 0; k < 16; ++k) printf("%.0f ", (double)out[k]);
    printf("\n");
}

int main(void) {
    printf("=== probe_d: decisive decomposition ===\n");
    store_check();
    per_block(128, 1);
    per_block(5120, 10);
    distinct_blocks();
    return 0;
}
