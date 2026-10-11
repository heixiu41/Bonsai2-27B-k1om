/* pq2prim.c -- KNC (k1om) primitive cost table for the PQ2_0 decoder work.
 *
 * WHY THIS EXISTS
 * ---------------
 * Every optimisation plan for this decoder is a bet on what one instruction
 * costs relative to another, and this project has already been burned twice by
 * guessing. The doc's decomposition ("vpermd = 62% of the time") is inferred
 * from whole-kernel deltas (V1 vs V4 vs V5) which mix the permutes, the
 * multiplies, the loads and the store/load round trip. This file measures the
 * primitives DIRECTLY: one instruction repeated on independent registers, so
 * the ratios are not contaminated by anything else.
 *
 * METHOD / TRAPS AVOIDED
 *   - NC independent chains (default 8) => the number is THROUGHPUT, not
 *     latency. Single-chain latency probes are included for vpermd/fma/add so
 *     the two can be told apart.
 *   - Every result is consumed into a volatile sink => nothing can be dead
 *     code eliminated (TRAP 1).
 *   - vpaddd/vpandd are the "one cheap ALU op" reference: a 512-bit integer
 *     add/and is 1 uop on KNC, so their ps/op IS the core clock period, and
 *     every other row can be read as a cycle count.
 *   - The disassembly (icc -mmic -O3 -S) is checked to confirm the loop
 *     contains the instruction mix it claims (TRAP 2).
 *
 * Card: Intel Xeon Phi 7120P, 61 cores / 244 threads, single thread here.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define NC 8

static volatile float g_fsink;
static volatile int   g_isink;

/* All constants are declared inside the bench macro; unused ones are folded
 * away by the compiler. a[k] is always __m512i and float ops bitcast. */
#define CVARS                                                                 \
    const __m512i c_mask = _mm512_set1_epi32(0x03030303);                     \
    const __m512i c_idx  = _mm512_set_epi32(15,14,13,12,11,10,9,8,             \
                                            7,6,5,4,3,2,1,0);                 \
    const __m512i c_sh   = _mm512_set1_epi32(6);                              \
    const __m512i c_k    = _mm512_set1_epi32((int)0x4B000000u);               \
    const __m512  c_f    = _mm512_set1_ps(1.0009765625f);                     \
    const __m512  c_f2   = _mm512_set1_ps(0.5f);                              \
    (void)c_mask; (void)c_idx; (void)c_sh; (void)c_k; (void)c_f; (void)c_f2

#define DEF_BENCH(name, body)                                                 \
static double name(long iters) {                                              \
    __m512i a[NC];                                                            \
    CVARS;                                                                    \
    for (int k = 0; k < NC; ++k) a[k] = _mm512_set1_epi32(k + 1);             \
    const double t0 = now_s();                                                \
    for (long i = 0; i < iters; ++i) {                                        \
        _Pragma("unroll")                                                     \
        for (int k = 0; k < NC; ++k) { body }                                 \
    }                                                                         \
    const double t1 = now_s();                                                \
    __m512i s = _mm512_setzero_si512();                                       \
    for (int k = 0; k < NC; ++k) s = _mm512_xor_si512(s, a[k]);               \
    g_isink = _mm512_reduce_add_epi32(s);                                     \
    return (t1 - t0) / ((double)iters * NC) * 1e12;                           \
}

#define PS(x)      _mm512_castsi512_ps(x)
#define SI(x)      _mm512_castps_si512(x)

DEF_BENCH(b_add,   { a[k] = _mm512_add_epi32(a[k], c_mask); })
DEF_BENCH(b_and,   { a[k] = _mm512_and_si512(a[k], c_mask); })
DEF_BENCH(b_or,    { a[k] = _mm512_or_si512(a[k], c_mask); })
DEF_BENCH(b_xor,   { a[k] = _mm512_xor_si512(a[k], c_mask); })
DEF_BENCH(b_srli,  { a[k] = _mm512_srli_epi32(a[k], 2); })
DEF_BENCH(b_srlv,  { a[k] = _mm512_srlv_epi32(a[k], c_sh); })
DEF_BENCH(b_sllv,  { a[k] = _mm512_sllv_epi32(a[k], c_sh); })
DEF_BENCH(b_mulld, { a[k] = _mm512_mullo_epi32(a[k], c_mask); })
/* chain through the INDEX operand: result = c_idx[a[k] & 15] */
DEF_BENCH(b_permd_idx, { a[k] = _mm512_permutevar_epi32(a[k], c_idx); })
/* chain through the DATA operand: result = a[k][c_idx & 15] */
DEF_BENCH(b_permd_dat, { a[k] = _mm512_permutevar_epi32(c_idx, a[k]); })
/* the materialisation this investigation is about: (c & mask) | K */
DEF_BENCH(b_and_or, { a[k] = _mm512_or_si512(_mm512_and_si512(a[k], c_mask), c_k); })
DEF_BENCH(b_and_or_sub, {
    a[k] = SI(_mm512_sub_ps(PS(_mm512_or_si512(
               _mm512_and_si512(a[k], c_mask), c_k)), c_f2));
})
DEF_BENCH(b_and_or_mul, {
    a[k] = SI(_mm512_mul_ps(PS(_mm512_or_si512(
               _mm512_and_si512(a[k], c_mask), c_k)), c_f));
})
DEF_BENCH(b_mulps, { a[k] = SI(_mm512_mul_ps(PS(a[k]), c_f)); })
DEF_BENCH(b_addps, { a[k] = SI(_mm512_add_ps(PS(a[k]), c_f)); })
DEF_BENCH(b_subps, { a[k] = SI(_mm512_sub_ps(PS(a[k]), c_f)); })
DEF_BENCH(b_fma,   { a[k] = SI(_mm512_fmadd_ps(PS(a[k]), c_f, c_f2)); })
/* the full 4-op V10c group body minus the permute: srli+or+and is folded */
DEF_BENCH(b_v10_g1, { a[k] = _mm512_srli_epi32(a[k], 2); })
DEF_BENCH(b_v10_g2, { a[k] = _mm512_permutevar_epi32(a[k], c_idx); })

/* ---- single-chain latency probes --------------------------------------- */
static double l_add(long iters) {
    __m512i a = _mm512_set1_epi32(1);
    const __m512i b = _mm512_set1_epi32(3);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) a = _mm512_add_epi32(a, b);
    const double t1 = now_s();
    g_isink = _mm512_reduce_add_epi32(a);
    return (t1 - t0) / (double)iters * 1e12;
}
static double l_permd(long iters) {
    __m512i a = _mm512_set1_epi32(1);
    const __m512i c = _mm512_set1_epi32(3);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) a = _mm512_permutevar_epi32(a, c);
    const double t1 = now_s();
    g_isink = _mm512_reduce_add_epi32(a);
    return (t1 - t0) / (double)iters * 1e12;
}
static double l_fma(long iters) {
    __m512 a = _mm512_set1_ps(1.0f);
    const __m512 b = _mm512_set1_ps(1.0000001f), c = _mm512_set1_ps(0.5f);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) a = _mm512_fmadd_ps(a, b, c);
    const double t1 = now_s();
    g_fsink = _mm512_reduce_add_ps(a);
    return (t1 - t0) / (double)iters * 1e12;
}
static double l_mulld(long iters) {
    __m512i a = _mm512_set1_epi32(3);
    const __m512i b = _mm512_set1_epi32(0x01010101);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) a = _mm512_mullo_epi32(a, b);
    const double t1 = now_s();
    g_isink = _mm512_reduce_add_epi32(a);
    return (t1 - t0) / (double)iters * 1e12;
}

/* ---- memory benches ---------------------------------------------------- */
#define NLOAD 8
static double load_ps_bench(long iters, const float *p) {
    __m512 a[NLOAD];
    for (int k = 0; k < NLOAD; ++k) a[k] = _mm512_setzero_ps();
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NLOAD; ++k)
            a[k] = _mm512_add_ps(a[k], _mm512_load_ps(p + 16 * k));
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int k = 0; k < NLOAD; ++k) s = _mm512_add_ps(s, a[k]);
    g_fsink = _mm512_reduce_add_ps(s);
    return (t1 - t0) / ((double)iters * NLOAD) * 1e12;
}

#define PQ2_IMCI_LOAD16(p) \
    _mm512_extloadunpackhi_epi32( \
        _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), \
            (const void *)(p), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE), \
        (const void *)((const uint8_t *)(p) + 64), \
        _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)

static double load16_bench(long iters, const uint8_t *p, size_t stride) {
    __m512i a[4];
    for (int k = 0; k < 4; ++k) a[k] = _mm512_setzero_si512();
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < 4; ++k)
            a[k] = _mm512_xor_si512(a[k], PQ2_IMCI_LOAD16(p + (size_t)k * stride));
    }
    const double t1 = now_s();
    __m512i s = _mm512_setzero_si512();
    for (int k = 0; k < 4; ++k) s = _mm512_xor_si512(s, a[k]);
    g_isink = _mm512_reduce_add_epi32(s);
    return (t1 - t0) / ((double)iters * 4) * 1e12;
}

int main(int argc, char **argv) {
    const long iters = argc > 1 ? atol(argv[1]) : 1000000L;

    printf("=== KNC primitive cost table (pq2prim) ===\n");
    printf("single thread, NC=8 independent chains, %ld iterations each\n", iters);
    printf("vpaddd/vpandd are the 'one cheap ALU op' reference: their ps/op IS\n");
    printf("the core clock period, so every other row is also a cycle count.\n\n");

    float *fb = (float *)aligned_alloc(64, 4096);
    uint8_t *bb = (uint8_t *)aligned_alloc(64, 4096);
    if (!fb || !bb) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (int i = 0; i < 1024; ++i) fb[i] = (float)((i % 7) - 3) * 0.25f;
    for (int i = 0; i < 4096; ++i) bb[i] = (uint8_t)(i * 37 + 11);

    const double ps_add  = b_add(iters);
    const double ps_and  = b_and(iters);
    const double ps_or   = b_or(iters);
    const double ps_xor  = b_xor(iters);
    const double ps_srli = b_srli(iters);
    const double ps_srlv = b_srlv(iters);
    const double ps_sllv = b_sllv(iters);
    const double ps_mulld= b_mulld(iters);
    const double ps_pidx = b_permd_idx(iters);
    const double ps_pdat = b_permd_dat(iters);
    const double ps_andor= b_and_or(iters);
    const double ps_aorsub = b_and_or_sub(iters);
    const double ps_aormul = b_and_or_mul(iters);
    const double ps_mulps= b_mulps(iters);
    const double ps_addps= b_addps(iters);
    const double ps_subps= b_subps(iters);
    const double ps_fma  = b_fma(iters);
    const double ps_load = load_ps_bench(iters, fb);
    const double ps_load16 = load16_bench(iters, bb + 2, 34);
    const double ps_lat_add  = l_add(iters);
    const double ps_lat_permd= l_permd(iters / 4 > 0 ? iters / 4 : 1);
    const double ps_lat_mulld= l_mulld(iters / 4 > 0 ? iters / 4 : 1);
    const double ps_lat_fma  = l_fma(iters);

    const double ref = ps_add;   /* one 512-bit integer add = 1 cycle */

    printf("%-34s %10s %9s\n", "primitive", "ps/op", "cycles");
    #define ROW(nm, v) printf("  %-34s %8.1f %9.2f\n", nm, (v), (v) / ref)
    ROW("vpaddd           (reference)", ps_add);
    ROW("vpandd", ps_and);
    ROW("vpord", ps_or);
    ROW("vpxord", ps_xor);
    ROW("vpsrld imm (srli)", ps_srli);
    ROW("vpsrlvd (srlv)", ps_srlv);
    ROW("vpsllvd (sllv)", ps_sllv);
    ROW("vpmulld", ps_mulld);
    ROW("vpermd  (chain thru index)", ps_pidx);
    ROW("vpermd  (chain thru data)", ps_pdat);
    ROW("and+or          (2 ops)", ps_andor);
    ROW("and+or+subps    (3 ops)", ps_aorsub);
    ROW("and+or+mulps    (3 ops)", ps_aormul);
    ROW("vmulps", ps_mulps);
    ROW("vaddps", ps_addps);
    ROW("vsubps", ps_subps);
    ROW("vfmadd (213)", ps_fma);
    ROW("vmovaps L1 64B x8", ps_load);
    ROW("LOAD16 = lo+hi unpack (2 instr)", ps_load16);
    ROW("--- latency, 1 chain ---", 0.0);
    ROW("vpaddd chain", ps_lat_add);
    ROW("vpermd chain", ps_lat_permd);
    ROW("vpmulld chain", ps_lat_mulld);
    ROW("vfmadd chain", ps_lat_fma);
    #undef ROW

    printf("\nimplied core clock if vpaddd == 1 cycle: %.1f ps = %.3f GHz\n",
           ps_add, 1e3 / ps_add * 1e6 / 1e6);
    printf("\nKEY RATIOS (the numbers every redesign bet depends on)\n");
    printf("  vpermd / vpaddd         = %6.2f   cheap INT ops per permute\n",
           ps_pdat / ref);
    printf("  vpermd / (and+or)       = %6.2f   vs the bias-float materialisation\n",
           ps_pdat / ps_andor);
    printf("  vpermd / (and+or+subps) = %6.2f\n", ps_pdat / ps_aorsub);
    printf("  (vpermd+vmulps)/(and+or+subps) = %6.2f\n",
           (ps_pdat + ps_mulps) / ps_aorsub);
    printf("  vfmadd / vpaddd         = %6.2f\n", ps_fma / ref);
    printf("  LOAD16 / vpaddd         = %6.2f\n", ps_load16 / ref);
    printf("  vmovaps / vpaddd        = %6.2f\n", ps_load / ref);
    printf("\nsink %d %g\n", g_isink, (double)g_fsink);
    free(fb); free(bb);
    return 0;
}
