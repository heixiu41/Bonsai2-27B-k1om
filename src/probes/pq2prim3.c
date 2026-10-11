/* pq2prim3.c -- decisive KNC issue-rate model.
 *
 * pq2prim/pq2prim2 both measured ~2 cycles for EVERY 512-bit vector op
 * (vpaddd, vpandd, vpermd, vpmulld, vmulps, vfmadd) with NC=8 independent
 * chains.  Two explanations survive and they lead to OPPOSITE design
 * conclusions, so they must be separated:
 *
 *   (H1) the core retires at most one 512-bit vector op per 2 cycles
 *        => decoder cost == instruction COUNT, and any replacement of a
 *           vpermd by 2 cheap ops is a LOSS.
 *   (H2) each measured op had its second operand in MEMORY (icc folded the
 *        constants into .L_2il0floatpacket.*(%rip)), so the table actually
 *        measured L1 bandwidth (one 64-byte operand fetch per 2 cycles)
 *        => register-resident / immediate ops are cheaper, and load-heavy
 *           kernels (LOAD16, broadcast permutes) have a different cost.
 *
 * This file tests H1 vs H2 by running the SAME operation with a register
 * operand, an immediate (no operand at all), and a memory operand, and by
 * scaling the number of independent chains 8 -> 16 (if the cost per op is an
 * issue-rate limit it stays constant; if it is a loop/front-end artifact it
 * changes).
 *
 * All results are consumed into a volatile sink.  The disassembly is checked
 * separately (srccount.py) to confirm each loop really contains the intended
 * operand form.
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

static volatile float g_fsink;
static volatile int   g_isink;

/* ---- clock calibration (4 dependent scalar adds / iteration) ------------ */
static double clock_hz(long iters) {
    long x = 1;
    const long one = 1;
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) { x += one; x += one; x += one; x += one; }
    const double t1 = now_s();
    g_isink = (int)x;
    return (double)iters * 4.0 / (t1 - t0);
}

/* ---- empty loop: the loop-overhead floor -------------------------------- */
__attribute__((noinline)) static double b_empty(long iters) {
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) { __asm__ __volatile__(""); }
    const double t1 = now_s();
    return (t1 - t0) / (double)iters * 1e12;
}

/* ------------------------------------------------------------------------
 * FMA with a REGISTER operand (no memory at all).
 * The constants are held in a vector that also depends on a runtime value, so
 * the compiler cannot fold them into a RIP-relative memory operand.
 * ---------------------------------------------------------------------- */
#define NCV 8
__attribute__((noinline)) static double b_fma_reg(long iters, __m512 *params) {
    __m512 a[NCV];
    const __m512 bb = params[0], cc = params[1];
    for (int k = 0; k < NCV; ++k) a[k] = _mm512_set1_ps((float)k + 1.0f);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NCV; ++k) a[k] = _mm512_fmadd_ps(a[k], bb, cc);
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int k = 0; k < NCV; ++k) s = _mm512_add_ps(s, a[k]);
    g_fsink = _mm512_reduce_add_ps(s);
    return (t1 - t0) / ((double)iters * NCV) * 1e12;
}

/* 16 independent chains, register operands: does ps/op stay the same? */
__attribute__((noinline)) static double b_fma_reg16(long iters, __m512 *params) {
    __m512 a[16];
    const __m512 bb = params[0], cc = params[1];
    for (int k = 0; k < 16; ++k) a[k] = _mm512_set1_ps((float)k + 1.0f);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < 16; ++k) a[k] = _mm512_fmadd_ps(a[k], bb, cc);
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int k = 0; k < 16; ++k) s = _mm512_add_ps(s, a[k]);
    g_fsink = _mm512_reduce_add_ps(s);
    return (t1 - t0) / ((double)iters * 16) * 1e12;
}

/* FMA with one MEMORY operand (the form pq2prim2 measured) */
__attribute__((noinline)) static double b_fma_mem(long iters, __m512 *params) {
    __m512 a[NCV];
    const __m512 bb = params[0];
    for (int k = 0; k < NCV; ++k) a[k] = _mm512_set1_ps((float)k + 1.0f);
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NCV; ++k) a[k] = _mm512_fmadd_ps(a[k], bb, params[2]);
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int k = 0; k < NCV; ++k) s = _mm512_add_ps(s, a[k]);
    g_fsink = _mm512_reduce_add_ps(s);
    return (t1 - t0) / ((double)iters * NCV) * 1e12;
}

/* ---- immediate-only operand: vpsrld $2 --------------------------------- */
__attribute__((noinline)) static double b_srli(long iters, __m512i *params) {
    __m512i a[NCV];
    for (int k = 0; k < NCV; ++k) a[k] = params[k];
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NCV; ++k) a[k] = _mm512_srli_epi32(a[k], 2);
    }
    const double t1 = now_s();
    __m512i s = _mm512_setzero_si512();
    for (int k = 0; k < NCV; ++k) s = _mm512_xor_si512(s, a[k]);
    g_isink = _mm512_reduce_add_epi32(s);
    return (t1 - t0) / ((double)iters * NCV) * 1e12;
}

/* ---- vpermd with all-register operands --------------------------------- */
__attribute__((noinline)) static double b_permd_reg(long iters, __m512i *params) {
    __m512i a[NCV];
    const __m512i idx = params[0];
    for (int k = 0; k < NCV; ++k) a[k] = params[k + 1];
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NCV; ++k) a[k] = _mm512_permutevar_epi32(idx, a[k]);
    }
    const double t1 = now_s();
    __m512i s = _mm512_setzero_si512();
    for (int k = 0; k < NCV; ++k) s = _mm512_xor_si512(s, a[k]);
    g_isink = _mm512_reduce_add_epi32(s);
    return (t1 - t0) / ((double)iters * NCV) * 1e12;
}

/* ---- vpaddd with all-register operands (two live registers) ------------ */
__attribute__((noinline)) static double b_add_reg(long iters, __m512i *params) {
    __m512i a[NCV], b[NCV];
    for (int k = 0; k < NCV; ++k) { a[k] = params[k]; b[k] = params[k + 1]; }
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NCV; ++k) a[k] = _mm512_add_epi32(a[k], b[k]);
    }
    const double t1 = now_s();
    __m512i s = _mm512_setzero_si512();
    for (int k = 0; k < NCV; ++k) s = _mm512_xor_si512(s, a[k]);
    g_isink = _mm512_reduce_add_epi32(s);
    return (t1 - t0) / ((double)iters * NCV) * 1e12;
}

/* ---- the MIX that matters: 4 int ops + 4 fma per iteration, registers -- */
__attribute__((noinline)) static double b_mix4(long iters, __m512i *params) {
    __m512i a[4];
    const __m512i m = params[0], kk = params[1];
    const __m512  f = _mm512_castsi512_ps(params[2]);
    __m512  s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512  s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int k = 0; k < 4; ++k) a[k] = params[k + 3];
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        __m512i t0i = _mm512_or_si512(_mm512_and_si512(a[0], m), kk);
        __m512i t1i = _mm512_or_si512(_mm512_and_si512(a[1], m), kk);
        __m512i t2i = _mm512_or_si512(_mm512_and_si512(a[2], m), kk);
        __m512i t3i = _mm512_or_si512(_mm512_and_si512(a[3], m), kk);
        s0 = _mm512_fmadd_ps(_mm512_castsi512_ps(t0i), f, s0);
        s1 = _mm512_fmadd_ps(_mm512_castsi512_ps(t1i), f, s1);
        s2 = _mm512_fmadd_ps(_mm512_castsi512_ps(t2i), f, s2);
        s3 = _mm512_fmadd_ps(_mm512_castsi512_ps(t3i), f, s3);
    }
    const double t1 = now_s();
    g_fsink = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                 _mm512_add_ps(s2, s3)));
    return (t1 - t0) / ((double)iters * 8) * 1e12;   /* 8 ops per iteration */
}

/* ---- the SAME mix but with a vpermd instead of and+or + mul by f ------- */
__attribute__((noinline)) static double b_mix4_perm(long iters, __m512i *params) {
    __m512i a[4];
    const __m512i idx = params[0], lut = params[1];
    const __m512  f = _mm512_castsi512_ps(params[2]);
    __m512  s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512  s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int k = 0; k < 4; ++k) a[k] = params[k + 3];
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        __m512i t0i = _mm512_permutevar_epi32(idx, _mm512_permutevar_epi32(a[0], lut));
        __m512i t1i = _mm512_permutevar_epi32(idx, _mm512_permutevar_epi32(a[1], lut));
        __m512i t2i = _mm512_permutevar_epi32(idx, _mm512_permutevar_epi32(a[2], lut));
        __m512i t3i = _mm512_permutevar_epi32(idx, _mm512_permutevar_epi32(a[3], lut));
        s0 = _mm512_fmadd_ps(_mm512_castsi512_ps(t0i), f, s0);
        s1 = _mm512_fmadd_ps(_mm512_castsi512_ps(t1i), f, s1);
        s2 = _mm512_fmadd_ps(_mm512_castsi512_ps(t2i), f, s2);
        s3 = _mm512_fmadd_ps(_mm512_castsi512_ps(t3i), f, s3);
    }
    const double t1 = now_s();
    g_fsink = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                 _mm512_add_ps(s2, s3)));
    return (t1 - t0) / ((double)iters * 12) * 1e12;  /* 12 ops per iteration */
}

int main(int argc, char **argv) {
    const long iters = argc > 1 ? atol(argv[1]) : 1000000L;
    printf("=== pq2prim3: H1 (issue-rate limit) vs H2 (memory-operand limit) ===\n");
    printf("single thread, %ld iterations\n\n", iters);

    static __m512  psp[8] __attribute__((aligned(64)));
    static __m512i psi[8] __attribute__((aligned(64)));
    for (int k = 0; k < 8; ++k) {
        psp[k] = _mm512_set1_ps(1.0f + 0.001f * (float)(k + 1));
        psi[k] = _mm512_set1_epi32(0x03030303 * (k + 1));
    }
    psi[0] = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    const double hz = clock_hz(iters / 4);
    const double cyc = 1e12 / hz;
    printf("clock = %.4f GHz  (1 cycle = %.1f ps)\n\n", hz / 1e9, cyc);

    const double ps_empty = b_empty(iters * 10);
    const double ps_fmareg = b_fma_reg(iters, psp);
    const double ps_fmareg16 = b_fma_reg16(iters, psp);
    const double ps_fmamem = b_fma_mem(iters, psp);
    const double ps_srli = b_srli(iters, psi);
    const double ps_permdreg = b_permd_reg(iters, psi);
    const double ps_addreg = b_add_reg(iters, psi);
    const double ps_mix4 = b_mix4(iters, psi);
    const double ps_mix4p = b_mix4_perm(iters, psi);

    printf("%-44s %10s %8s %10s\n", "loop", "ps/op", "cycles", "ps/iter");
    #define ROW(nm, v, nops) printf("  %-42s %10.1f %8.2f %10.1f\n", nm, (v), (v)/cyc, (v)*(nops))
    ROW("empty loop (overhead/iter)", ps_empty, 1);
    ROW("vfmadd, REGISTER operands, 8 chains", ps_fmareg, 8);
    ROW("vfmadd, REGISTER operands, 16 chains", ps_fmareg16, 16);
    ROW("vfmadd, MEMORY operand, 8 chains", ps_fmamem, 8);
    ROW("vpsrld $2 (immediate only)", ps_srli, 8);
    ROW("vpermd, REGISTER operands", ps_permdreg, 8);
    ROW("vpaddd, REGISTER operands", ps_addreg, 8);
    ROW("mix4: and+or+fma x4 (8 ops/iter)", ps_mix4, 8);
    ROW("mix4p: 2*vpermd+fma x4 (12 ops/iter)", ps_mix4p, 12);
    #undef ROW

    printf("\nINTERPRETATION\n");
    printf("  empty loop overhead = %.1f ps/iteration (%.2f cycles)\n",
           ps_empty, ps_empty / cyc);
    printf("  vfmadd reg 8 chains  = %.2f cycles/op\n", ps_fmareg / cyc);
    printf("  vfmadd reg 16 chains = %.2f cycles/op\n", ps_fmareg16 / cyc);
    printf("  vfmadd mem 8 chains  = %.2f cycles/op\n", ps_fmamem / cyc);
    printf("  vpsrld immediate     = %.2f cycles/op\n", ps_srli / cyc);
    printf("  vpermd register      = %.2f cycles/op\n", ps_permdreg / cyc);
    printf("  vpaddd register      = %.2f cycles/op\n", ps_addreg / cyc);
    printf("\n  mix4  (4x(and+or+fma)) = %.2f cycles/op  (%.1f cycles/iteration)\n",
           ps_mix4 / cyc, ps_mix4 * 8 / cyc);
    printf("  mix4p (4x(2 perm+fma)) = %.2f cycles/op  (%.1f cycles/iteration)\n",
           ps_mix4p / cyc, ps_mix4p * 12 / cyc);
    printf("  => per 16 weights: bias-float design %.2f cyc, "
           "LUT-permute design %.2f cyc\n",
           ps_mix4 * 8 / cyc, ps_mix4p * 12 / cyc);
    printf("\nsink %d %g\n", g_isink, (double)g_fsink);
    return 0;
}
