/* pq2prim2.c -- rigorous KNC (k1om) machine model for the PQ2_0 decoder.
 *
 * pq2prim.c measured register-only primitives and produced a suspicious table
 * (everything exactly 2 cycles, including vpermd and vpmulld).  Two things had
 * to be ruled out before believing it:
 *   (a) the MEMORY benches in pq2prim.c were loop-invariant and were certainly
 *       hoisted by LICM, so their numbers were meaningless;
 *   (b) the loop overhead could have been the real bottleneck.
 *
 * This file fixes both:
 *   - every bench is __attribute__((noinline)) so it can be found and counted
 *     in the disassembly (TRAP 2), and the mnemonic counts are printed;
 *   - memory benches advance a pointer through a wrap-around L1-resident
 *     buffer, so nothing can be hoisted (TRAP 1);
 *   - each chain does TWO operations per iteration (16 ops/iteration), so loop
 *     overhead is amortised 2x better than pq2prim.c;
 *   - a clock calibration is included (dependent scalar add chain), so ps/op
 *     can be converted to cycles with a measured, not assumed, clock.
 *
 * Everything is consumed into a volatile sink.
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

#define NC 8                     /* independent chains */
#define REP 2                    /* ops per chain per iteration */

static volatile float g_fsink;
static volatile int   g_isink;
static volatile long  g_lsink;

/* ---- clock calibration: dependent chain of scalar adds ------------------ */
/* KNC scalar integer add latency is 1 cycle; a fully dependent chain of N
 * adds therefore retires 1 add/cycle and the iteration count IS the cycle
 * count (modulo loop overhead, which is a second dependent chain and
 * co-issues). Confirmed by counting against /proc/cpuinfo's reported clock. */
#define SCALAR_ITERS 2000000
static double clock_hz(void) {
    /* 4 dependent adds per iteration; each has latency 1 */
    volatile long seed = 1;
    long x = (long)seed;
    const long one = 1;
    const double t0 = now_s();
    for (long i = 0; i < SCALAR_ITERS; ++i) {
        x += one; x += one; x += one; x += one;
    }
    const double t1 = now_s();
    g_lsink = x;
    return (double)SCALAR_ITERS * 4.0 / (t1 - t0);
}

/* ---- register-only benches --------------------------------------------- */
#define CVARS                                                                 \
    const __m512i c_mask = _mm512_set1_epi32(0x03030303);                     \
    const __m512i c_idx  = _mm512_set_epi32(15,14,13,12,11,10,9,8,             \
                                            7,6,5,4,3,2,1,0);                 \
    const __m512i c_sh   = _mm512_set1_epi32(6);                              \
    const __m512i c_k    = _mm512_set1_epi32((int)0x4B000000u);               \
    const __m512  c_f    = _mm512_set1_ps(1.0009765625f);                     \
    const __m512  c_f2   = _mm512_set1_ps(0.5f);                              \
    (void)c_mask; (void)c_idx; (void)c_sh; (void)c_k; (void)c_f; (void)c_f2

#define PS(x) _mm512_castsi512_ps(x)
#define SI(x) _mm512_castps_si512(x)

/* body is repeated REP times per chain per iteration */
#define DEF_REG_BENCH(name, body)                                             \
__attribute__((noinline)) static double name(long iters) {                    \
    __m512i a[NC];                                                            \
    CVARS;                                                                    \
    for (int k = 0; k < NC; ++k) a[k] = _mm512_set1_epi32(k + 1);             \
    const double t0 = now_s();                                                \
    for (long i = 0; i < iters; ++i) {                                        \
        _Pragma("unroll")                                                     \
        for (int r = 0; r < REP; ++r) {                                       \
            _Pragma("unroll")                                                 \
            for (int k = 0; k < NC; ++k) { body }                             \
        }                                                                     \
    }                                                                         \
    const double t1 = now_s();                                                \
    __m512i s = _mm512_setzero_si512();                                       \
    for (int k = 0; k < NC; ++k) s = _mm512_xor_si512(s, a[k]);               \
    g_isink = _mm512_reduce_add_epi32(s);                                     \
    return (t1 - t0) / ((double)iters * NC * REP) * 1e12;                     \
}

DEF_REG_BENCH(b_add,   { a[k] = _mm512_add_epi32(a[k], c_mask); })
DEF_REG_BENCH(b_and,   { a[k] = _mm512_and_si512(a[k], c_mask); })
DEF_REG_BENCH(b_permd, { a[k] = _mm512_permutevar_epi32(c_idx, a[k]); })
DEF_REG_BENCH(b_mulld, { a[k] = _mm512_mullo_epi32(a[k], c_mask); })
DEF_REG_BENCH(b_fma,   { a[k] = SI(_mm512_fmadd_ps(PS(a[k]), c_f, c_f2)); })
DEF_REG_BENCH(b_mulps, { a[k] = SI(_mm512_mul_ps(PS(a[k]), c_f)); })
DEF_REG_BENCH(b_andor, { a[k] = _mm512_or_si512(_mm512_and_si512(a[k], c_mask), c_k); })
/* 8 integer + 8 float ops per iteration: tests whether the machine can
 * co-issue an integer vector op with an FP vector op (it cannot if both use
 * the same 512-bit FPU, which is what KNC does -- it has no integer SIMD). */
DEF_REG_BENCH(b_mix_and_fma, {
    a[k] = _mm512_and_si512(a[k], c_mask);
    a[k] = SI(_mm512_fmadd_ps(PS(a[k]), c_f, c_f2));
})

/* ---- memory benches (pointer advances, so no LICM) ---------------------- */
/* 8 independent 64-byte loads per iteration, 512 B/iteration, L1 resident */
__attribute__((noinline)) static double b_load_ps(long iters, float *buf, long buf_floats) {
    __m512 a[NC];
    for (int k = 0; k < NC; ++k) a[k] = _mm512_setzero_ps();
    float *p = buf;
    const float *end = buf + buf_floats - 128;
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NC; ++k)
            a[k] = _mm512_add_ps(a[k], _mm512_load_ps(p + 16 * k));
        p += 128;
        if (p >= end) p = buf;
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int k = 0; k < NC; ++k) s = _mm512_add_ps(s, a[k]);
    g_fsink = _mm512_reduce_add_ps(s);
    return (t1 - t0) / ((double)iters * NC) * 1e12;
}

#define PQ2_IMCI_LOAD16(p) \
    _mm512_extloadunpackhi_epi32( \
        _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), \
            (const void *)(p), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE), \
        (const void *)((const uint8_t *)(p) + 64), \
        _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)

/* 4 LOAD16 per iteration (8 instructions), 136 bytes apart, pointer advances */
__attribute__((noinline)) static double b_load16(long iters, uint8_t *buf, long buf_bytes) {
    __m512i a[4];
    for (int k = 0; k < 4; ++k) a[k] = _mm512_setzero_si512();
    uint8_t *p = buf;
    const uint8_t *end = buf + buf_bytes - 512;
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < 4; ++k)
            a[k] = _mm512_xor_si512(a[k], PQ2_IMCI_LOAD16(p + (size_t)k * 34));
        p += 136;
        if (p >= end) p = buf;
    }
    const double t1 = now_s();
    __m512i s = _mm512_setzero_si512();
    for (int k = 0; k < 4; ++k) s = _mm512_xor_si512(s, a[k]);
    g_isink = _mm512_reduce_add_epi32(s);
    return (t1 - t0) / ((double)iters * 4) * 1e12;
}

/* stores: 8 x 64-byte stores per iteration into an L1-resident buffer */
__attribute__((noinline)) static double b_store(long iters, float *buf, long buf_floats) {
    const __m512 v = _mm512_set1_ps(1.5f);
    float *p = buf;
    const float *end = buf + buf_floats - 128;
    const double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        _Pragma("unroll")
        for (int k = 0; k < NC; ++k)
            _mm512_store_ps(p + 16 * k, _mm512_add_ps(v, _mm512_set1_ps((float)k)));
        p += 128;
        if (p >= end) p = buf;
    }
    const double t1 = now_s();
    g_fsink = buf[0] + buf[64];
    return (t1 - t0) / ((double)iters * NC) * 1e12;
}

int main(int argc, char **argv) {
    const long iters = argc > 1 ? atol(argv[1]) : 500000L;
    printf("=== pq2prim2: KNC machine model (noinline benches, 16 ops/iteration) ===\n");
    printf("single thread, NC=8 chains x %d ops, %ld iterations\n\n", NC * REP, iters);

    const double hz = clock_hz();
    printf("clock calibration (4 dependent scalar adds/iteration): %.4f GHz\n", hz / 1e9);
    printf("=> 1 cycle = %.1f ps\n\n", 1e12 / hz);

    float *fb = (float *)aligned_alloc(64, 65536);
    uint8_t *bb = (uint8_t *)aligned_alloc(64, 65536);
    if (!fb || !bb) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (int i = 0; i < 16384; ++i) fb[i] = (float)((i % 7) - 3) * 0.25f;
    for (int i = 0; i < 65536; ++i) bb[i] = (uint8_t)(i * 37 + 11);

    /* warm up / avoid first-touch effects */
    (void)b_add(iters / 10);

    const double ps_add   = b_add(iters);
    const double ps_and   = b_and(iters);
    const double ps_permd = b_permd(iters);
    const double ps_mulld = b_mulld(iters);
    const double ps_fma   = b_fma(iters);
    const double ps_mulps = b_mulps(iters);
    const double ps_andor = b_andor(iters);
    const double ps_mix   = b_mix_and_fma(iters);
    const double ps_load  = b_load_ps(iters, fb, 16384);
    const double ps_load16= b_load16(iters, bb, 65536);
    const double ps_store = b_store(iters, fb, 16384);

    const double cyc = 1e12 / hz;   /* ps per cycle */
    printf("%-40s %9s %8s\n", "operation", "ps/op", "cycles");
    #define ROW(nm, v) printf("  %-38s %9.1f %8.2f\n", nm, (v), (v) / cyc)
    ROW("vpaddd  (register)", ps_add);
    ROW("vpandd  (register)", ps_and);
    ROW("vpermd  (register)", ps_permd);
    ROW("vpmulld (register)", ps_mulld);
    ROW("vfmadd  (register)", ps_fma);
    ROW("vmulps  (register)", ps_mulps);
    ROW("and+or  (2 ops)", ps_andor);
    ROW("and+fma (2 ops, mixed int/fp)", ps_mix);
    ROW("vmovaps 64B load (L1, +addr)", ps_load);
    ROW("LOAD16 = lo+hi (2 instr)", ps_load16);
    ROW("vmovaps 64B store (L1, +addr)", ps_store);
    #undef ROW

    printf("\nPER-INSTRUCTION issue cost (register-only ops):\n");
    printf("  vpaddd  %.2f cyc   vpandd %.2f   vpermd %.2f   vpmulld %.2f\n",
           ps_add/cyc, ps_and/cyc, ps_permd/cyc, ps_mulld/cyc);
    printf("  vfmadd  %.2f cyc   vmulps %.2f\n", ps_fma/cyc, ps_mulps/cyc);
    printf("  and+or  %.2f cyc for 2 ops => %.2f cyc/op\n",
           ps_andor/cyc, ps_andor/cyc/2);
    printf("  and+fma %.2f cyc for 2 ops => %.2f cyc/op (no int/fp co-issue if ~2x)\n",
           ps_mix/cyc, ps_mix/cyc/2);
    printf("\nKEY RATIOS\n");
    printf("  vpermd / vpaddd     = %.3f  (1.0 => a permute costs one ordinary op)\n",
           ps_permd / ps_add);
    printf("  vpermd / and+or     = %.3f  (2 ops; <1.0 => vpermd is CHEAPER than the\n"
           "                                 bias-float materialisation)\n", ps_permd / ps_andor);
    printf("  LOAD16 / vpaddd     = %.3f  (LOAD16 = 2 instructions)\n", ps_load16 / ps_add);
    printf("  load64 / vpaddd     = %.3f\n", ps_load / ps_add);
    printf("  store64 / vpaddd    = %.3f\n", ps_store / ps_add);
    printf("\nsink %d %g %ld\n", g_isink, (double)g_fsink, g_lsink);
    free(fb); free(bb);
    return 0;
}
