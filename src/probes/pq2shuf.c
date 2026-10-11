/* Are KNC's 128-bit-LANE-LOCAL shuffles cheaper than the full cross-lane vpermd?
 *
 * WHY THIS MATTERS. A sibling probe measured every 512-bit op at ~1.65 ns and
 * concluded "instruction count is the only currency". But its list covered only
 * vpaddd/vpandd/vpsrld/vpmulld/vpermd/vmulps/vsubps/vfmadd/vscaleps -- NOT the
 * lane-local shuffles. Those may well execute in fewer cycles because they do
 * not need the crossbar:
 *
 *   _mm512_swizzle_epi32(v, imm)      -- permute dwords WITHIN each 128-bit lane
 *   _mm512_permute4f128_epi32(v, imm) -- permute the four 128-bit lanes
 *   _mm512_shuffle_epi32(v, imm)      -- same as swizzle (AVX-512 name)
 *
 * If any of them is ~2x cheaper than vpermd, the PQ2_0 decoder's LUT gather can
 * be restructured to use them instead, because the 16-entry LUT is indexed by
 * only 2 bits of information per 16-weight group.
 *
 * METHOD (respecting the two traps that produced fiction earlier):
 *   - 8 independent chains so latency is hidden and THROUGHPUT is measured;
 *   - inner loop unrolled by macro, each op feeding the next accumulation;
 *   - result folded into a volatile sink that is PRINTED;
 *   - the per-op cost is derived from the same harness for every candidate, so
 *     only the RATIO matters (absolute ns/op is not the point).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

static volatile int isink;
static volatile float fsink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define NC   8
#define ITER 2000000

#define BENCH(NAME, EXPR)                                                   \
    do {                                                                    \
        __m512i a[NC]; __m512 f[NC];                                        \
        for (int i = 0; i < NC; ++i) { a[i] = _mm512_set1_epi32(i + 1);     \
                                       f[i] = _mm512_set1_ps((float)i + 1);}\
        const __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0); \
        const double t0 = now_s();                                          \
        for (int it = 0; it < ITER; ++it) {                                 \
            for (int i = 0; i < NC; ++i) { EXPR; }                          \
        }                                                                   \
        const double t1 = now_s();                                          \
        double s = 0;                                                       \
        for (int i = 0; i < NC; ++i) s += _mm512_reduce_add_ps(_mm512_add_ps( \
              _mm512_castsi512_ps(a[i]), f[i]));                            \
        fsink += (float)s; isink += (int)s;                                 \
        const double ns = (t1 - t0) / ((double)ITER * NC) * 1e9;            \
        printf("  %-34s %8.3f ns/op\n", NAME, ns);                          \
    } while (0)

/* one op that feeds itself; the assignment back to a[i] creates a chain, and
 * NC=8 chains give enough ILP to reach throughput */
#define CHAIN_I(dst, expr) dst = (expr)

int main(void) {
    printf("=== KNC 512-bit op cost: is a lane-local shuffle cheaper? ===\n");
    printf("NC=%d independent chains, ITER=%d, clockcheck below\n\n", NC, ITER);

    /* sanity: clock resolution / monotonicity */
    {
        const double a = now_s(); double b = a;
        int spins = 0;
        while ((b - a) < 1e-6 && spins < 1000000) { b = now_s(); ++spins; }
        printf("clock: resolved 1us after %d spins -> %s\n\n",
               spins, b > a ? "monotonic OK" : "BROKEN");
    }

    BENCH("vpaddd               (baseline)",
          CHAIN_I(a[i], _mm512_add_epi32(a[i], idx)));
    BENCH("vpermd               (full cross-lane)",
          CHAIN_I(a[i], _mm512_permutevar_epi32(idx, a[i])));
    BENCH("swizzle_epi32        (128-bit lane local)",
          CHAIN_I(a[i], _mm512_swizzle_epi32(a[i], _MM_SWIZ_REG_BADC)));
    BENCH("shuffle_epi32        (128-bit lane local)",
          CHAIN_I(a[i], _mm512_shuffle_epi32(a[i], _MM_PERM_BADC)));
    BENCH("permute4f128_epi32   (lane granularity)",
          CHAIN_I(a[i], _mm512_permute4f128_epi32(a[i], _MM_PERM_BADC)));
    BENCH("vpandd               (baseline cheap)",
          CHAIN_I(a[i], _mm512_and_si512(a[i], idx)));
    BENCH("vpsrld imm",
          CHAIN_I(a[i], _mm512_srli_epi32(a[i], 2)));
    BENCH("vpsrlvd var",
          CHAIN_I(a[i], _mm512_srlv_epi32(a[i], idx)));
    BENCH("vfmadd_ps",
          CHAIN_I(f[i], _mm512_fmadd_ps(f[i], _mm512_set1_ps(1.0f), f[i])));
    BENCH("vmulps",
          CHAIN_I(f[i], _mm512_mul_ps(f[i], _mm512_set1_ps(1.0f))));

    printf("\nsink %d %.3f\n", (int)isink, (float)fsink);
    printf("\nIf swizzle/shuffle/permute4f128 are materially cheaper than vpermd,\n");
    printf("the decoder's LUT gather can be rebuilt on them.\n");
    return 0;
}
