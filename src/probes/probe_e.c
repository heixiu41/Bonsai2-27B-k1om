/* probe_e.c -- smallest possible reproduction: one block, one FMA round.
 *
 * probe_d proved (on the card):
 *   - _mm512_store_ps writes all 16 lanes exactly
 *   - reduce == store+scalar for every accumulator
 *   - a single 128-weight block of -1 gives -64 total where the four
 *     accumulators are each -16 and the sum must be -128
 *   - with distinct per-block weights the kernel returns exactly HALF of every
 *     block's contribution (-52480 = -104960/2), not a subset of blocks
 *
 * So the loss is inside one 128-weight block's four FMAs. This probe prints the
 * exact value of each accumulator BEFORE and AFTER each FMA, and follows the
 * same register from FMA through store and through reduce.
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

static void dump16(const char *tag, __m512 v) {
    float s[16] __attribute__((aligned(64)));
    _mm512_store_ps(s, v);
    printf("      %-14s:", tag);
    for (int k = 0; k < 16; ++k) printf(" %6.1f", (double)s[k]);
    printf("   reduce=%8.1f\n", (double)_mm512_reduce_add_ps(v));
}

int main(void) {
    const int n = 128;                 /* exactly one block */
    const int nblk = 1;
    uint8_t *w = (uint8_t *)al((size_t)nblk * PQ2_0_BLOCK_BYTES);
    float   *x = (float *)al((size_t)n * sizeof(float));
    w[0] = 0x00; w[1] = 0x3C;          /* fp16 1.0 */
    memset(w + 2, 0x00, 32);           /* every code 0 -> weight -1.0 */
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    printf("=== probe_e: one block, one FMA round, full trace ===\n");
    printf("  expected: every weight -1.0, x all 1.0, y = -128\n\n");

    float wf[QK_PQ2_0] __attribute__((aligned(64)));
    pq2_0_dequant_block(w, wf);
    printf("  wf[] after dequant:\n    ");
    for (int k = 0; k < 128; ++k) {
        printf("%.0f", (double)wf[k]);
        if (k != 127) printf("%s", ((k & 15) == 15) ? "\n    " : " ");
    }
    printf("\n");
    {
        double s = 0;
        for (int k = 0; k < 128; ++k) s += (double)wf[k];
        printf("  scalar sum(wf) = %.1f\n\n", s);
    }

    printf("  x[0..3] = %.1f %.1f %.1f %.1f   x[64..67] = %.1f %.1f %.1f %.1f\n\n",
           (double)x[0], (double)x[1], (double)x[2], (double)x[3],
           (double)x[64], (double)x[65], (double)x[66], (double)x[67]);

    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    dump16("sum0 before", sum0);
    dump16("sum1 before", sum1);
    dump16("sum2 before", sum2);
    dump16("sum3 before", sum3);

    __m512 p0 = _mm512_mul_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + 0));
    __m512 p1 = _mm512_mul_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + 16));
    __m512 p2 = _mm512_mul_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + 32));
    __m512 p3 = _mm512_mul_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + 48));
    printf("\n  --- the four products before accumulation ---\n");
    dump16("p0", p0); dump16("p1", p1); dump16("p2", p2); dump16("p3", p3);

    sum0 = _mm512_add_ps(p0, sum0);
    sum1 = _mm512_add_ps(p1, sum1);
    sum2 = _mm512_add_ps(p2, sum2);
    sum3 = _mm512_add_ps(p3, sum3);
    printf("\n  --- the four accumulators after ONE add (should each be -16) ---\n");
    dump16("sum0", sum0); dump16("sum1", sum1);
    dump16("sum2", sum2); dump16("sum3", sum3);

    const double r0 = (double)_mm512_reduce_add_ps(sum0);
    const double r1 = (double)_mm512_reduce_add_ps(sum1);
    const double r2 = (double)_mm512_reduce_add_ps(sum2);
    const double r3 = (double)_mm512_reduce_add_ps(sum3);
    printf("\n  --- reductions ---\n");
    printf("      reduce(sum0..3) = %.1f %.1f %.1f %.1f   scalar total = %.1f\n",
           r0, r1, r2, r3, r0 + r1 + r2 + r3);
    printf("      correct scalar total = -128.0\n");

    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    dump16("sum0+1+2+3", sum);
    printf("      reduce(add tree) = %.1f   (correct -128.0)\n",
           (double)_mm512_reduce_add_ps(sum));

    printf("\n  --- now the same thing with fmadd, as the kernel does ---\n");
    {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        a0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + 0),  a0);
        a1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + 48), a3);
        dump16("fmadd a0", a0); dump16("fmadd a1", a1);
        dump16("fmadd a2", a2); dump16("fmadd a3", a3);
        __m512 t = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
        dump16("fmadd total", t);
        printf("      reduce = %.1f  (correct -128.0)\n",
               (double)_mm512_reduce_add_ps(t));
    }

    /* the same, but reading x through a pointer the compiler cannot fold */
    printf("\n  --- mul+add, x read through a volatile pointer ---\n");
    {
        const float * volatile xv = x;
        __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
        __m512 b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
        b0 = _mm512_add_ps(b0, _mm512_mul_ps(_mm512_load_ps(wf),      _mm512_load_ps(xv + 0)));
        b1 = _mm512_add_ps(b1, _mm512_mul_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(xv + 16)));
        b2 = _mm512_add_ps(b2, _mm512_mul_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(xv + 32)));
        b3 = _mm512_add_ps(b3, _mm512_mul_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(xv + 48)));
        __m512 t = _mm512_add_ps(_mm512_add_ps(b0, b1), _mm512_add_ps(b2, b3));
        dump16("volatile total", t);
        printf("      reduce = %.1f  (correct -128.0)\n",
               (double)_mm512_reduce_add_ps(t));
    }
    free(w); free(x);
    return 0;
}
