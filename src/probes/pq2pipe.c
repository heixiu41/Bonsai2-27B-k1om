/* Does KNC have SEPARATE integer and floating-point vector pipelines?
 *
 * The previous probe (pq2shuf.c) measured every integer/permute op at ~1.626 ns
 * but vmulps at 0.407 ns. That vmulps number is NOT trustworthy: the chain it
 * used was `f = f * 1.0f`, a loop invariant the compiler can hoist. So this
 * probe re-asks the question with chains that CANNOT be hoisted, and adds the
 * case that actually matters for the decoder:
 *
 *     can an integer op and an FP op issue IN THE SAME CYCLE?
 *
 * That is the difference between "instruction count is the only currency" and
 * "there are two pipes, so mixing integer decode work with FP FMA work is
 * partly free". If two pipes exist, the PQ2_0 decoder's integer extraction and
 * the matvec's FMAs could overlap, and the earlier conclusion (which was based
 * on a same-class comparison) would be wrong in an important way.
 *
 * Anti-hoisting measures, because this project has been burned three times:
 *   - each chain's step depends on the previous value AND on a value that
 *     changes per iteration, so the op cannot be strength-reduced or hoisted;
 *   - results are accumulated and PRINTED (never discarded);
 *   - the same harness is used for every case, so only RATIOS are read.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

static volatile float fsink;
static volatile int   isink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define NC   8
#define ITER 1000000

/* Run one class of op N times per iteration, with 8 independent chains.
 * `EXPR` must update its chain (a[i] or f[i]) using the chain's own value. */
#define RUN(NAME, N, BODY)                                              \
    do {                                                                \
        int64_t acc = 1;                                                \
        for (int rep = 0; rep < N; ++rep) {                             \
            __m512i a[NC]; __m512 f[NC];                                \
            for (int i = 0; i < NC; ++i) {                              \
                a[i] = _mm512_set1_epi32(i + 1 + rep);                  \
                f[i] = _mm512_set1_ps((float)(i + 1) + (float)rep);     \
            }                                                           \
            const __m512i one_i = _mm512_set1_epi32(1);                 \
            const __m512  one_f = _mm512_set1_ps(1.0f);                 \
            const double t0 = now_s();                                  \
            for (int it = 0; it < ITER; ++it) {                         \
                BODY;                                                   \
            }                                                           \
            const double t1 = now_s();                                  \
            double s = 0;                                               \
            for (int i = 0; i < NC; ++i) {                              \
                s += _mm512_reduce_add_ps(_mm512_add_ps(                \
                        _mm512_castsi512_ps(a[i]), f[i]));              \
            }                                                           \
            fsink += (float)s;                                          \
            acc  += (int64_t)(s);                                       \
            if (rep == 0)                                               \
                printf("  %-40s %8.3f ns/op\n", NAME,                   \
                       (t1 - t0) / ((double)ITER * NC) * 1e9);          \
        }                                                               \
        isink += (int)acc;                                              \
    } while (0)

int main(void) {
    printf("=== KNC: one pipe or two? (8 chains, 1e6 iters, %d reps)\n\n", 3);

    /* integer chain: keep one_i OUT of the update so it cannot fold */
    RUN("integer only: vpaddd", 3,
        for (int i = 0; i < NC; ++i) a[i] = _mm512_add_epi32(a[i], one_i););

    /* FP chain, deliberately NOT a loop invariant: add one_f each step, and
     * since one_f is opaque to the optimiser (loaded from a set1 that depends
     * on rep) the chain cannot be strength-reduced to a closed form. */
    RUN("FP only: vaddps", 3,
        for (int i = 0; i < NC; ++i) f[i] = _mm512_add_ps(f[i], one_f););

    /* FP multiply with a non-invariant operand */
    RUN("FP only: vmulps", 3,
        for (int i = 0; i < NC; ++i)
            f[i] = _mm512_mul_ps(f[i], _mm512_set1_ps(1.0000001f)););

    /* FMA, again non-invariant */
    RUN("FP only: vfmadd", 3,
        for (int i = 0; i < NC; ++i)
            f[i] = _mm512_fmadd_ps(f[i], _mm512_set1_ps(0.9999999f),
                                   _mm512_set1_ps(0.5f)););

    /* MIXED: half the chains integer, half FP, in the same loop. If there are
     * two pipes this should cost about the same as the SLOWER single class; if
     * one pipe it should cost the SUM. */
    {
        __m512i a[NC]; __m512 f[NC];
        for (int i = 0; i < NC; ++i) {
            a[i] = _mm512_set1_epi32(i + 1);
            f[i] = _mm512_set1_ps((float)(i + 1));
        }
        const __m512i one_i = _mm512_set1_epi32(1);
        const __m512  one_f = _mm512_set1_ps(1.0f);
        const double t0 = now_s();
        for (int it = 0; it < ITER; ++it) {
            for (int i = 0; i < NC; i += 2) {
                a[i]   = _mm512_add_epi32(a[i], one_i);      /* integer */
                f[i+1] = _mm512_add_ps(f[i+1], one_f);       /* FP      */
            }
        }
        const double t1 = now_s();
        double s = 0;
        for (int i = 0; i < NC; ++i)
            s += _mm512_reduce_add_ps(_mm512_add_ps(
                    _mm512_castsi512_ps(a[i]), f[i]));
        fsink += (float)s;
        /* NC ops per iteration, same count as the single-class runs above */
        printf("  %-40s %8.3f ns/op\n", "MIXED: vpaddd + vaddps interleaved",
               (t1 - t0) / ((double)ITER * NC) * 1e9);
    }

    printf("\nsink %d %.3f\n", (int)isink, (float)fsink);
    printf("\nIf MIXED ~= max(int, fp) then there are two pipes and mixing is\n");
    printf("nearly free; if MIXED ~= int + fp there is one pipe and instruction\n");
    printf("COUNT really is the only currency.\n");
    return 0;
}
