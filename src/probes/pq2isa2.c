/* pq2isa2.c -- (a) measure the SEMANTICS and (b) measure the THROUGHPUT of the
 * k1om instructions that pq2isa_scan.ps1 proved are actually emitted (not just
 * declared).  Everything observable: results are printed AND folded into a
 * volatile sink so nothing can be dead-code-eliminated (Trap 1).
 *
 * Throughput protocol: 8 independent dependency chains, `iters` iterations,
 * one clock_gettime pair around the loop.  8 chains means the reported value is
 * a THROUGHPUT (not a latency) whenever latency < 8 * throughput.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

static volatile int   g_sink_i;
static volatile float g_sink_f;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void pdw(const char *tag, __m512i v) {
    int t[16] __attribute__((aligned(64)));
    _mm512_store_epi32(t, v);
    printf("  %-16s", tag);
    for (int i = 0; i < 16; ++i) printf(" %11d", t[i]);
    printf("\n");
}
static void pfl(const char *tag, __m512 v) {
    float t[16] __attribute__((aligned(64)));
    _mm512_store_ps(t, v);
    printf("  %-16s", tag);
    for (int i = 0; i < 16; ++i) printf(" %11.4g", t[i]);
    printf("\n");
}
static void pxw(const char *tag, __m512i v) {
    int t[16] __attribute__((aligned(64)));
    _mm512_store_epi32(t, v);
    printf("  %-16s", tag);
    for (int i = 0; i < 16; ++i) printf(" %08x", (unsigned)t[i]);
    printf("\n");
}

/* ===================== semantics ====================================== */
static void semantics(void) {
    int ii[16] __attribute__((aligned(64)));
    float ff[16] __attribute__((aligned(64)));
    uint8_t  b8[64]  __attribute__((aligned(64)));
    uint16_t h16[32] __attribute__((aligned(64)));
    int32_t  x32[32] __attribute__((aligned(64)));
    for (int i = 0; i < 16; ++i) { ii[i] = i; ff[i] = (float)i; }
    for (int i = 0; i < 64; ++i) b8[i] = (uint8_t)(i * 3 + 1);
    for (int i = 0; i < 32; ++i) h16[i] = (uint16_t)(0x3C00 + i);   /* 1.0,1.0009.. */
    for (int i = 0; i < 32; ++i) x32[i] = 1000 + i;

    printf("=== semantics ===\n");
    printf("inputs: ii = 0..15, ff = 0..15, b8[i]=3i+1, h16[i]=0x3C00+i (fp16 1.0+), x32[i]=1000+i\n\n");

    /* vfmadd233ps : what is the multiplier/addend pattern? */
    printf("[vfmadd233ps]  a=ii  b=ff+16 (i.e. b[i]=16+i)\n");
    {
        __m512 a = _mm512_load_ps(ff);                       /* 0..15   */
        __m512 b = _mm512_add_ps(_mm512_load_ps(ff), _mm512_set1_ps(16.0f));
        pfl("a", a); pfl("b", b);
        pfl("fmadd233(a,b)", _mm512_fmadd233_ps(a, b));
        pfl("fmadd233(b,a)", _mm512_fmadd233_ps(b, a));
        pfl("a*b+2^23 (ref)", _mm512_add_ps(_mm512_mul_ps(a, b), _mm512_set1_ps(8388608.0f)));
    }
    /* vpmadd233d */
    printf("\n[vpmadd233d]  a=0..15  b=1000+i\n");
    {
        __m512i a = _mm512_load_epi32(ii);
        __m512i b = _mm512_load_epi32(x32);
        pdw("a", a); pdw("b", b);
        pdw("madd233(a,b)", _mm512_fmadd233_epi32(a, b));
        pdw("madd233(b,a)", _mm512_fmadd233_epi32(b, a));
    }
    /* vscaleps */
    printf("\n[vscaleps]  a=1.5, b=0..15 -> expect 1.5*2^b\n");
    {
        __m512 a = _mm512_set1_ps(1.5f);
        __m512i b = _mm512_load_epi32(ii);
        pfl("scale(a,b)", _mm512_scale_ps(a, b));
    }
    /* extload_ps with upconverts */
    printf("\n[extload_ps UINT8]  b8=3i+1 -> expect 3i+1 as float\n");
    pfl("u8", _mm512_extload_ps(b8, _MM_UPCONV_PS_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
    printf("[extload_ps FLOAT16] h16=0x3C00+i (=1.0 + i*2^-10)\n");
    pfl("f16", _mm512_extload_ps(h16, _MM_UPCONV_PS_FLOAT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
    printf("[extload_ps SINT16] x32=1000+i\n");
    pfl("s16", _mm512_extload_ps(x32, _MM_UPCONV_PS_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
    printf("[extload_epi32 UINT8]\n");
    pdw("u8", _mm512_extload_epi32(b8, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
    printf("[extloadunpacklo_epi32 UINT8]  (dst=zero)\n");
    pdw("lo", _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), b8,
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE));
    pdw("hi", _mm512_extloadunpackhi_epi32(_mm512_setzero_si512(), b8,
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE));
    printf("[extload_ps UINT8 with 4X16 broadcast]\n");
    pfl("bc4x16", _mm512_extload_ps(b8, _MM_UPCONV_PS_UINT8, _MM_BROADCAST_4X16, _MM_HINT_NONE));
    pfl("bc1x16", _mm512_extload_ps(b8, _MM_UPCONV_PS_UINT8, _MM_BROADCAST_1X16, _MM_HINT_NONE));

    /* permutes */
    printf("\n[permutes]  data = 0..15 (dword lanes)\n");
    {
        __m512i d = _mm512_load_epi32(ii);
        pdw("data", d);
        pdw("perm4f128 0x1B", _mm512_permute4f128_epi32(d, 0x1B));
        pdw("perm4f128 0xE4", _mm512_permute4f128_epi32(d, 0xE4));
        pdw("perm4f128 0x00", _mm512_permute4f128_epi32(d, 0x00));
        pdw("vpshufd 0x1B", _mm512_shuffle_epi32(d, (_MM_PERM_ENUM)0x1B));
        pdw("valignd(d,d,3)", _mm512_alignr_epi32(d, d, 3));
        __m512i e = _mm512_add_epi32(d, _mm512_set1_epi32(100));
        pdw("valignd(d,e,3)", _mm512_alignr_epi32(d, e, 3));
    }
    /* swizzle variants -- the enum must be an immediate */
    {
        __m512i d = _mm512_load_epi32(ii);
        __m512  f = _mm512_load_ps(ff);
        pdw("swz NONE",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_NONE));
        pdw("swz CDAB",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_CDAB));
        pdw("swz BADC",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_BADC));
        pdw("swz AAAA",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_AAAA));
        pdw("swz BBBB",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_BBBB));
        pdw("swz CCCC",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_CCCC));
        pdw("swz DDDD",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_DDDD));
        pdw("swz DACB",  _mm512_swizzle_epi32(d, _MM_SWIZ_REG_DACB));
        pfl("swz_ps CDAB", _mm512_swizzle_ps(f, _MM_SWIZ_REG_CDAB));
        pfl("swz_ps AAAA", _mm512_swizzle_ps(f, _MM_SWIZ_REG_AAAA));
        pfl("swz_ps DACB", _mm512_swizzle_ps(f, _MM_SWIZ_REG_DACB));
    }
    /* high multiplies / set-mask arithmetic */
    printf("\n[mulhi]\n");
    {
        __m512i a = _mm512_set1_epi32(0x40000000);
        pdw("mulhd(a,a)", _mm512_mulhi_epi32(a, a));
        pdw("mulhud(a,a)", _mm512_mulhi_epu32(a, a));
        pdw("mulld(a,a)", _mm512_mullo_epi32(a, a));
    }
    printf("\n[set-mask arith]  a=0..15, b=10\n");
    {
        __m512i a = _mm512_load_epi32(ii);
        __m512i b = _mm512_set1_epi32(10);
        __mmask16 k = 0;
        pdw("a", a);
        pdw("addsetc(a,b)", _mm512_addsetc_epi32(a, b, &k));
        printf("  addsetc mask = %04x\n", (unsigned)k);
        k = 0;
        pdw("subsetb(a,b)", _mm512_subsetb_epi32(a, b, &k));
        printf("  subsetb mask = %04x\n", (unsigned)k);
        k = 0;
        pfl("addsets_ps(a,b)", _mm512_addsets_ps(_mm512_load_ps(ff), _mm512_set1_ps(10.0f), &k));
        printf("  addsets mask = %04x\n", (unsigned)k);
    }
    /* gather */
    printf("\n[gather]  base = b8 (bytes 3i+1), offsets = i*2, scale=1\n");
    {
        __m512i off = _mm512_load_epi32(ii);
        pdw("gather_epi32 u16", _mm512_i32extgather_epi32(_mm512_slli_epi32(off, 1), b8,
                _MM_UPCONV_EPI32_UINT16, 1, _MM_HINT_NONE));
        pfl("gather_ps u8", _mm512_i32extgather_ps(off, b8,
                _MM_UPCONV_PS_UINT8, 1, _MM_HINT_NONE));
        pfl("gather_ps f16", _mm512_i32extgather_ps(_mm512_slli_epi32(off, 1), h16,
                _MM_UPCONV_PS_FLOAT16, 1, _MM_HINT_NONE));
    }
}

/* ===================== throughput ===================================== */
#define INIT0 _mm512_set1_epi32(0x0F0E0D0C)
#define C1   _mm512_set1_epi32(0x03030303)
#define CF1  _mm512_set1_ps(1.5f)
#define CF2  _mm512_set1_ps(0.25f)
#define SINK8() do { \
    __m512i s = _mm512_xor_si512(_mm512_xor_si512(a0,a1), _mm512_xor_si512(a2,a3)); \
    s = _mm512_xor_si512(s, _mm512_xor_si512(_mm512_xor_si512(a4,a5), _mm512_xor_si512(a6,a7))); \
    g_sink_i += _mm512_reduce_add_epi32(s); } while (0)

#define BENCHI(NAME, SETUP, STEP)                                        \
static double bench_##NAME(int iters) {                                  \
    SETUP;                                                               \
    __m512i a0=INIT0,a1=INIT0,a2=INIT0,a3=INIT0,a4=INIT0,a5=INIT0,a6=INIT0,a7=INIT0; \
    double t0 = now_s();                                                 \
    for (int i = 0; i < iters; ++i) {                                    \
        a0=STEP(a0); a1=STEP(a1); a2=STEP(a2); a3=STEP(a3);              \
        a4=STEP(a4); a5=STEP(a5); a6=STEP(a6); a7=STEP(a7);              \
    }                                                                    \
    double t1 = now_s();                                                 \
    SINK8();                                                             \
    return (t1 - t0) / (double)(iters * 8);                              \
}

#define F_LAST _mm512_castsi512_ps(_mm512_set1_epi32(0x3F800000))
#define BENCHF(NAME, SETUP, STEP)                                        \
static double bench_##NAME(int iters) {                                  \
    SETUP;                                                               \
    __m512 a0=F_LAST,a1=F_LAST,a2=F_LAST,a3=F_LAST,a4=F_LAST,a5=F_LAST,a6=F_LAST,a7=F_LAST; \
    double t0 = now_s();                                                 \
    for (int i = 0; i < iters; ++i) {                                    \
        a0=STEP(a0); a1=STEP(a1); a2=STEP(a2); a3=STEP(a3);              \
        a4=STEP(a4); a5=STEP(a5); a6=STEP(a6); a7=STEP(a7);              \
    }                                                                    \
    double t1 = now_s();                                                 \
    __m512i s = _mm512_castps_si512(_mm512_add_ps(_mm512_add_ps(a0,a1),  \
                          _mm512_add_ps(_mm512_add_ps(a2,a3),            \
                          _mm512_add_ps(_mm512_add_ps(a4,a5),            \
                                        _mm512_add_ps(a6,a7)))));        \
    g_sink_i += _mm512_reduce_add_epi32(s);                              \
    return (t1 - t0) / (double)(iters * 8);                              \
}

static uint8_t g_buf[4096] __attribute__((aligned(64)));

/* step macros: one per op under test */
#define S_addd(x)     _mm512_add_epi32(x, C1)
#define S_andd(x)     _mm512_and_si512(x, C1)
#define S_ord(x)      _mm512_or_si512(x, C1)
#define S_srli(x)     _mm512_srli_epi32(x, 2)
#define S_slli(x)     _mm512_slli_epi32(x, 2)
#define S_srlv(x)     _mm512_srlv_epi32(x, C1)
#define S_sllv(x)     _mm512_sllv_epi32(x, C1)
#define S_mulld(x)    _mm512_mullo_epi32(x, C1)
#define S_mulhd(x)    _mm512_mulhi_epi32(x, C1)
#define S_vpermd(x)   _mm512_permutevar_epi32(C1, x)
#define S_vpermd2(x)  _mm512_permutevar_epi32(x, C1)
#define S_permf32x4(x) _mm512_permute4f128_epi32(x, 0x1B)
#define S_vpshufd(x)  _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0x1B)
#define S_valignd(x)  _mm512_alignr_epi32(x, C1, 3)
#define S_swizzle(x)  _mm512_swizzle_epi32(x, (_MM_SWIZZLE_ENUM)1)
#define S_ldepi32(x)  _mm512_add_epi32(x, _mm512_load_epi32(g_buf))
#define S_ldextu8(x)  _mm512_add_epi32(x, _mm512_extload_epi32(g_buf,_MM_UPCONV_EPI32_UINT8,_MM_BROADCAST32_NONE,_MM_HINT_NONE))
#define S_ldexunpl(x) _mm512_add_epi32(x, _mm512_extloadunpacklo_epi32(x,g_buf,_MM_UPCONV_EPI32_UINT8,_MM_HINT_NONE))
#define S_cmpeq(x)    _mm512_mask_add_epi32(x, _mm512_cmpeq_epi32_mask(x, C1), x, C1)

BENCHI(addd,   (void)0, S_addd)
BENCHI(andd,   (void)0, S_andd)
BENCHI(ord,    (void)0, S_ord)
BENCHI(srli,   (void)0, S_srli)
BENCHI(slli,   (void)0, S_slli)
BENCHI(srlv,   (void)0, S_srlv)
BENCHI(sllv,   (void)0, S_sllv)
BENCHI(mulld,  (void)0, S_mulld)
BENCHI(mulhd,  (void)0, S_mulhd)
BENCHI(vpermd, (void)0, S_vpermd)
BENCHI(vpermd2,(void)0, S_vpermd2)
BENCHI(permf32x4, (void)0, S_permf32x4)
BENCHI(vpshufd, (void)0, S_vpshufd)
BENCHI(valignd, (void)0, S_valignd)
BENCHI(swizzle, (void)0, S_swizzle)
BENCHI(ldepi32, (void)0, S_ldepi32)
BENCHI(ldextu8, (void)0, S_ldextu8)
BENCHI(ldexunpl,(void)0, S_ldexunpl)
BENCHI(cmpeq,  (void)0, S_cmpeq)

#define S_addps(x)    _mm512_add_ps(x, CF1)
#define S_subps(x)    _mm512_sub_ps(x, CF1)
#define S_mulps(x)    _mm512_mul_ps(x, CF1)
#define S_fmadd(x)    _mm512_fmadd_ps(x, CF1, CF2)
#define S_scaleps(x)  _mm512_scale_ps(x, C1)
#define S_fmadd233(x) _mm512_fmadd233_ps(x, CF1)
#define S_maxps(x)    _mm512_max_ps(x, CF1)
#define S_gmaxabs(x)  _mm512_gmaxabs_ps(x, CF1)
#define S_ldps(x)     _mm512_add_ps(x, _mm512_load_ps(g_buf))
#define S_ldextpsu8(x) _mm512_add_ps(x, _mm512_extload_ps(g_buf,_MM_UPCONV_PS_UINT8,_MM_BROADCAST32_NONE,_MM_HINT_NONE))
#define S_ldextpsf16(x) _mm512_add_ps(x, _mm512_extload_ps(g_buf,_MM_UPCONV_PS_FLOAT16,_MM_BROADCAST32_NONE,_MM_HINT_NONE))
#define S_gatherps(x) _mm512_add_ps(x, _mm512_i32extgather_ps(_mm512_and_si512(_mm512_set1_epi32(0x3F), C1), g_buf, _MM_UPCONV_PS_NONE, 1, _MM_HINT_NONE))

BENCHF(addps,  (void)0, S_addps)
BENCHF(subps,  (void)0, S_subps)
BENCHF(mulps,  (void)0, S_mulps)
BENCHF(fmadd,  (void)0, S_fmadd)
BENCHF(scaleps,(void)0, S_scaleps)
BENCHF(fmadd233,(void)0, S_fmadd233)
BENCHF(maxps,  (void)0, S_maxps)
BENCHF(gmaxabs,(void)0, S_gmaxabs)
BENCHF(ldps,   (void)0, S_ldps)
BENCHF(ldextpsu8,(void)0, S_ldextpsu8)
BENCHF(ldextpsf16,(void)0, S_ldextpsf16)
BENCHF(gatherps,(void)0, S_gatherps)

static void throughput(int iters);

int main(int argc, char **argv) {
    const int mode  = argc > 1 ? atoi(argv[1]) : 0;
    const int iters = argc > 2 ? atoi(argv[2]) : 200000;
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 0; i < 4096; ++i) g_buf[i] = (uint8_t)(i * 7 + 3);
    /* make the load sources deterministic */
    memcpy(g_buf, "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10", 16);

    if (mode == 0) { throughput(iters); return 0; }
    if (mode == 1) { semantics(); return 0; }
    throughput(iters);
    semantics();
    return 0;
}

static void throughput(int iters) {
    printf("=== throughput (8 independent chains, %d iterations) ===\n", iters);
    printf("ns per instruction; 1 core clock ~= 0.808 ns at 1.238 GHz\n");
#define RUNI(n) printf("  %-14s %8.3f ns/op\n", #n, bench_##n(iters)*1e9)
#define RUNF(n) printf("  %-14s %8.3f ns/op   [float]\n", #n, bench_##n(iters)*1e9)
    RUNI(addd); RUNI(andd); RUNI(ord); RUNI(srli); RUNI(slli);
    RUNI(srlv); RUNI(sllv); RUNI(mulld); RUNI(mulhd);
    RUNI(vpermd); RUNI(vpermd2); RUNI(permf32x4); RUNI(vpshufd);
    RUNI(valignd); RUNI(swizzle);
    RUNI(ldepi32); RUNI(ldextu8); RUNI(ldexunpl); RUNI(cmpeq);
    printf("  ---\n");
    RUNF(addps); RUNF(subps); RUNF(mulps); RUNF(fmadd); RUNF(scaleps);
    RUNF(fmadd233); RUNF(maxps); RUNF(gmaxabs);
    RUNF(ldps); RUNF(ldextpsu8); RUNF(ldextpsf16); RUNF(gatherps);
    printf("sink %d %g\n", g_sink_i, (double)g_sink_f);
}
