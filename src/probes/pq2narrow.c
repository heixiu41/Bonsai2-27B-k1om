/* Narrow down the IMCI "exactly half" defect.
 *
 * Established facts (pq2diag.c, on the card):
 *   - one block, reduce(load(wf + g*16)) for g=0..3 each == -16  (16 lanes, correct)
 *   - scalar dequant of 5120 weights == -5120                    (correct)
 *   - but the 4 x 16-lane accumulators over 40 blocks give
 *       s0=s1=s2=s3=-640  where -1280 is expected
 *
 * Hypotheses this program distinguishes:
 *   H1  _mm512_reduce_add_ps only sums part of the vector
 *   H2  _mm512_fmadd_ps behaves as a narrower operation on this part
 *   H3  the loop body executes only half as often as expected
 *   H4  the accumulator vectors alias / are not preserved across iterations
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
    float a[16] __attribute__((aligned(64)));
    float b[16] __attribute__((aligned(64)));
    for (int i = 0; i < 16; ++i) { a[i] = 1.0f; b[i] = 1.0f; }

    printf("=== H1: what does _mm512_reduce_add_ps actually sum? ===\n");
    {
        __m512 va = _mm512_load_ps(a);
        printf("  reduce(add(load(ones)))      = %.1f   (expect 16)\n",
               _mm512_reduce_add_ps(va));

        /* distinct values so a partial reduction is visible */
        float c[16] __attribute__((aligned(64)));
        for (int i = 0; i < 16; ++i) c[i] = (float)(i + 1);   /* 1..16, sum 136 */
        __m512 vc = _mm512_load_ps(c);
        printf("  reduce(load(1..16))          = %.1f   (expect 136)\n",
               _mm512_reduce_add_ps(vc));

        /* explicit scalar sum of the same array for reference */
        double s = 0; for (int i = 0; i < 16; ++i) s += c[i];
        printf("  scalar sum(1..16)            = %.1f\n", s);

        /* sum the vector again after an explicit add, in case reduce is partial */
        printf("  (_mm512_add_ps identity twice = %.1f)\n",
               _mm512_reduce_add_ps(_mm512_add_ps(vc, _mm512_setzero_ps())));
    }

    printf("\n=== H2: is fmadd 16-wide?  ones*ones accumulated 4 times ===\n");
    {
        __m512 acc = _mm512_setzero_ps();
        __m512 one = _mm512_load_ps(a);
        for (int t = 0; t < 4; ++t) acc = _mm512_fmadd_ps(one, one, acc);
        printf("  reduce(acc) after 4 fmadd     = %.1f   (expect 64)\n",
               _mm512_reduce_add_ps(acc));

        /* same with mul+add instead of fmadd */
        __m512 acc2 = _mm512_setzero_ps();
        for (int t = 0; t < 4; ++t)
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(one, one));
        printf("  reduce(acc2) mul+add, 4 times = %.1f   (expect 64)\n",
               _mm512_reduce_add_ps(acc2));
    }

    printf("\n=== H3: how many times does the block loop body execute? ===\n");
    {
        const int n = 5120, nblk = n / QK_PQ2_0;
        int hits = 0;
        volatile float keep = 0.0f;
        for (int j = 0; j < n; j += QK_PQ2_0) {
            ++hits;
            keep += (float)j;
        }
        printf("  loop iterations              = %d   (expect %d)\n", hits, nblk);
        (void)keep;
    }

    printf("\n=== H4: does a 4-accumulator row reproduce the half value? ===\n");
    {
        const int n = 5120, nblk = n / QK_PQ2_0;
        const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
        uint8_t *w = (uint8_t *)aligned_alloc(64, (row_bytes + 63) & ~(size_t)63);
        float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
        for (int b = 0; b < nblk; ++b) {
            uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;             /* fp16 1.0 */
            memset(blk + 2, 0x00, 32);                /* codes 0 => -1.0 */
        }
        for (int i = 0; i < n; ++i) x[i] = 1.0f;

        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        int iterations = 0;
        for (int j = 0; j < n; j += QK_PQ2_0) {
            ++iterations;
            pq2_0_dequant_block(w + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            s0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      s0);
            s1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), s1);
            s2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), s2);
            s3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), s3);
        }
        printf("  iterations executed          = %d\n", iterations);
        printf("  s0=%.1f s1=%.1f s2=%.1f s3=%.1f   sum=%.1f   (expect -1280 each, -5120 total)\n",
               _mm512_reduce_add_ps(s0), _mm512_reduce_add_ps(s1),
               _mm512_reduce_add_ps(s2), _mm512_reduce_add_ps(s3),
               _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                  _mm512_add_ps(s2, s3))));

        /* same loop but accumulating with mul+add instead of fmadd */
        __m512 t0 = _mm512_setzero_ps(), t1 = _mm512_setzero_ps();
        __m512 t2 = _mm512_setzero_ps(), t3 = _mm512_setzero_ps();
        for (int j = 0; j < n; j += QK_PQ2_0) {
            pq2_0_dequant_block(w + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            t0 = _mm512_add_ps(t0, _mm512_mul_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j)));
            t1 = _mm512_add_ps(t1, _mm512_mul_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16)));
            t2 = _mm512_add_ps(t2, _mm512_mul_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32)));
            t3 = _mm512_add_ps(t3, _mm512_mul_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48)));
        }
        printf("  mul+add variant total        = %.1f   (expect -5120)\n",
               _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(t0, t1),
                                                  _mm512_add_ps(t2, t3))));
        free(w); free(x);
    }
    return 0;
}
