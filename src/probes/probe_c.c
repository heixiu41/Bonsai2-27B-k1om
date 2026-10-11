/* probe_c.c -- does _mm512_reduce_add_ps get constant-folded, and what does it
 * lower to when it cannot be folded?
 *
 * probe_b's reduction probe returned the CORRECT answer, but it reduced a vector
 * that came straight from a load of a const-filled array, so the compiler may
 * simply have computed the scalar sum at compile time and never emitted the
 * intrinsic at all. This probe makes the two cases explicit:
 *
 *   compute_foldable()  : reduce(load(const array))       -- foldable
 *   compute_live()      : reduce(vector built by inlined asm-free arithmetic
 *                         on volatile input)              -- not foldable
 *   compute_stored()    : store the SAME live vector and sum it scalar-wise
 *
 * Compile with -S and compare the three functions; run on the card to compare
 * live-reduce against stored-sum on identical data.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

static volatile float g_seed = 1.0f;

/* (1) foldable: the source vector is a load from a constant array */
float compute_foldable(void) {
    float v[16] __attribute__((aligned(64)));
    for (int k = 0; k < 16; ++k) v[k] = 1000.0f + (float)k;
    return _mm512_reduce_add_ps(_mm512_load_ps(v));
}

/* (2) live: build the vector by 16 independent adds so the value only exists
 *     as a vector register. volatile inputs stop constant folding. */
float compute_live(void) {
    float base[16] __attribute__((aligned(64)));
    for (int k = 0; k < 16; ++k) base[k] = 1000.0f + (float)k;
    __m512 a = _mm512_load_ps(base);
    __m512 acc = _mm512_setzero_ps();
    for (int t = 0; t < 5; ++t) {
        acc = _mm512_add_ps(acc, a);
        if (g_seed > 0.0f) acc = _mm512_add_ps(acc, _mm512_set1_ps(g_seed)); /* +1 each */
    }
    return _mm512_reduce_add_ps(acc);
}

/* (3) the same live vector written to memory and summed scalar-wise */
float compute_stored(void) {
    float base[16] __attribute__((aligned(64)));
    for (int k = 0; k < 16; ++k) base[k] = 1000.0f + (float)k;
    __m512 a = _mm512_load_ps(base);
    __m512 acc = _mm512_setzero_ps();
    for (int t = 0; t < 5; ++t) {
        acc = _mm512_add_ps(acc, a);
        if (g_seed > 0.0f) acc = _mm512_add_ps(acc, _mm512_set1_ps(g_seed));
    }
    float out[16] __attribute__((aligned(64)));
    _mm512_store_ps(out, acc);
    float s = 0.0f;
    for (int k = 0; k < 16; ++k) s += out[k];
    return s;
}

int main(void) {
    const float f = compute_foldable();
    const float l = compute_live();
    const float s = compute_stored();
    printf("=== probe_c: reduce folding / lowering ===\n");
    printf("  foldable : %.1f\n", (double)f);
    printf("  live     : %.1f  (reduce of a register vector)\n", (double)l);
    printf("  stored   : %.1f  (same vector, stored then summed)\n", (double)s);
    printf("  live vs stored: %s\n",
           (fabs((double)l - (double)s) < 0.5) ? "AGREE" : "*** DISAGREE ***");
    /* expected: base sum 16120 * 5 = 80600, plus 5 * 16 = 80 -> 80680 */
    printf("  expected : 80680.0\n");
    return 0;
}
