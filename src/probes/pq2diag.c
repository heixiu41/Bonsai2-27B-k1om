/* Isolate the exact half-loss in the IMCI PQ2_0 kernel.
 *
 * pq2mini established: the kernel returns exactly HALF the correct value for
 * every uniform pattern, and code==1 (all zero weights) gives the right answer.
 * That means half the accumulations never reach y[i].
 *
 * This program re-implements the kernel body step by step and reports the
 * partial sums, so we can see which of the four lanes is missing:
 *   - the 4 x 16 accumulation lanes
 *   - the four 64-float groups inside a 128-weight block
 *   - the 64-float chunk boundaries across the whole row
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "pq2_0.h"

int main(void) {
    const int n = 5120;
    const int nblk = n / QK_PQ2_0;              /* 40 */
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;

    uint8_t *w = (uint8_t *)aligned_alloc(64, (row_bytes + 63) & ~(size_t)63);
    float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    if (!w || !x) return 1;

    /* all codes = 0 (-1), scale = 1.0 -> every weight is exactly -1.0f */
    for (int b = 0; b < nblk; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;           /* fp16 1.0 */
        memset(blk + 2, 0x00, 32);              /* all codes 0 => -1 */
    }
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    printf("=== IMCI kernel step-by-step (n=%d, all weights = -1, x = 1) ===\n", n);
    printf("    correct answer is -%d\n\n", n);

    /* ---- step 1: raw scalar dequant sum, no SIMD ---- */
    {
        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        double tot = 0.0;
        for (int j = 0; j < n; j += QK_PQ2_0) {
            pq2_0_dequant_block(w + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            for (int k = 0; k < QK_PQ2_0; ++k) tot += wf[k] * x[j + k];
        }
        printf("  scalar dequant sum            = %10.1f   %s\n", tot,
               fabs(tot + n) < 1e-3 ? "correct" : "WRONG");
    }

    /* ---- step 2: the four accumulators, reported separately ---- */
    {
        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        for (int j = 0; j < n; j += QK_PQ2_0) {
            pq2_0_dequant_block(w + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            s0 = _mm512_add_ps(s0, _mm512_mul_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j)));
            s1 = _mm512_add_ps(s1, _mm512_mul_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16)));
            s2 = _mm512_add_ps(s2, _mm512_mul_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32)));
            s3 = _mm512_add_ps(s3, _mm512_mul_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48)));
        }
        const float r0 = _mm512_reduce_add_ps(s0), r1 = _mm512_reduce_add_ps(s1);
        const float r2 = _mm512_reduce_add_ps(s2), r3 = _mm512_reduce_add_ps(s3);
        printf("  lane sums: s0=%9.1f s1=%9.1f s2=%9.1f s3=%9.1f\n", r0, r1, r2, r3);
        printf("  s0+s1+s2+s3                   = %10.1f   %s\n",
               r0 + r1 + r2 + r3,
               fabs((r0 + r1 + r2 + r3) + n) < 1e-3 ? "correct" : "WRONG");
        printf("  sum of the four _mm512 vectors= %10.1f\n",
               (double)_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                          _mm512_add_ps(s2, s3))));
        printf("  NOTE expected each lane sum = %d (10 blocks x 16 lanes x -1)\n", -n/4);
    }

    /* ---- step 3: does _mm512_load_ps actually see the whole block? ---- */
    {
        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        pq2_0_dequant_block(w, wf);
        float lanetot[4];
        for (int g = 0; g < 4; ++g) {
            __m512 v = _mm512_load_ps(wf + g * 16);
            lanetot[g] = _mm512_reduce_add_ps(v);
        }
        printf("\n  one block: reduce(load(wf+0/16/32/48)) = %.1f %.1f %.1f %.1f  (each should be -16)\n",
               lanetot[0], lanetot[1], lanetot[2], lanetot[3]);
        printf("  scalar check of the same block       = ");
        double s = 0; for (int k = 0; k < QK_PQ2_0; ++k) s += wf[k];
        printf("%.1f\n", s);
        printf("  wf[0]=%.1f wf[64]=%.1f wf[127]=%.1f\n", wf[0], wf[64], wf[127]);
    }

    free(w); free(x);
    return 0;
}
