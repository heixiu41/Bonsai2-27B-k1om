/* probe_e2.c -- pin down the exact law of _mm512_reduce_add_ps on k1om.
 *
 * probe_e measured, for register-built (non-constant-foldable) vectors:
 *     reduce(a)                  where a = 16 x -1.0   -> -16   (correct -16)
 *     reduce(4 x a)              where 4 x a = 16 x -4 -> -64   (correct -64)
 *     reduce(p0+p1+p2+p3) with each p = 16 x -1        -> -64   (correct -128)
 *     reduce(sum0+sum1+sum2+sum3), each sum = 16 x -1  -> -64   (correct -128)
 * i.e. the last two are short by exactly 2x, while the first two are exact.
 *
 * Hypothesis: the intrinsic only ever handles the LOW 128-bit quarter and
 * effectively COUNTS it once per 128-bit lane, i.e.
 *     reduce(v)  ==  4 * (v[0] + v[1] + v[2] + v[3])
 * which is exact whenever all four quarters hold the same values (the case
 * every earlier test used) and wrong by exactly 2x when they do not.
 *
 * This probe tests the hypothesis directly with four DISTINCT quarters.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

static float sum16(__m512 v) {
    float s[16] __attribute__((aligned(64)));
    _mm512_store_ps(s, v);
    float t = 0.0f;
    for (int k = 0; k < 16; ++k) t += s[k];
    return t;
}
static float low4(__m512 v) {
    float s[16] __attribute__((aligned(64)));
    _mm512_store_ps(s, v);
    return s[0] + s[1] + s[2] + s[3];
}

static void check(const char *tag, __m512 v) {
    const float r = _mm512_reduce_add_ps(v);
    const float s = sum16(v);
    const float l = low4(v);
    const float l4 = 4.0f * l;
    printf("  %-22s reduce=%9.1f  true=%9.1f  4*low4=%9.1f  %s\n",
           tag, (double)r, (double)s, (double)l4,
           (fabsf(r - l4) < 0.01f) ? "reduce == 4*low4  <-- CONFIRMS" :
           (fabsf(r - s)  < 0.01f) ? "reduce == true (exact)" : "neither");
}

int main(void) {
    /* distinct quarters: quarter g holds 1000*(g+1) + k */
    float v[16] __attribute__((aligned(64)));
    for (int g = 0; g < 4; ++g)
        for (int k = 0; k < 4; ++k)
            v[g * 4 + k] = 1000.0f * (float)(g + 1) + (float)k;

    printf("=== probe_e2: exact law of _mm512_reduce_add_ps ===\n");
    printf("  quarters: q0=%s ... \n", "1000,1001,1002,1003");
    printf("  q0 sum=4006  q1 sum=4022  q2 sum=4038  q3 sum=4054  true total=16120\n\n");

    __m512 vec = _mm512_load_ps(v);
    check("distinct quarters", vec);
    printf("      (if reduce == 4*4006 == 16024 the intrinsic ignores q1,q2,q3)\n\n");

    /* duplicated-quarters controls, the case all earlier tests used */
    {
        float u[16] __attribute__((aligned(64)));
        for (int k = 0; k < 16; ++k) u[k] = -1.0f;
        check("all four quarters -1", _mm512_load_ps(u));
        for (int k = 0; k < 16; ++k) u[k] = -40.0f;
        check("all four quarters -40", _mm512_load_ps(u));
    }

    /* the exact shape probe_e hit: four 128-bit vectors each -1, then summed */
    {
        float u[16] __attribute__((aligned(64)));
        for (int k = 0; k < 16; ++k) u[k] = -1.0f;
        __m512 a = _mm512_load_ps(u);
        __m512 s = _mm512_add_ps(_mm512_add_ps(a, a), _mm512_add_ps(a, a));
        check("4 vectors each all -1", s);
        printf("      correct here is -64 (16 lanes at -4); probe_e measured -64,\n");
        printf("      so this case is EXACT and cannot be the defect.\n");
    }

    /* two DIFFERENT vectors, the shape that loses a factor of two */
    {
        float u[16] __attribute__((aligned(64)));   /* -1 */
        float t[16] __attribute__((aligned(64)));   /* -3 */
        for (int k = 0; k < 16; ++k) { u[k] = -1.0f; t[k] = -3.0f; }
        __m512 s = _mm512_add_ps(_mm512_load_ps(u), _mm512_load_ps(t));
        check("(-1 vec) + (-3 vec)", s);          /* true -64, 4*low4 also -64 */
    }
    return 0;
}
