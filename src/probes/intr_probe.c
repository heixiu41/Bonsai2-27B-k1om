/* intr_probe.c -- link + semantics probe for the KNC (k1om) IMCI intrinsics
 * needed by a fully vectorised PQ2_0 dequantiser.
 *
 * WHY: on k1om the header is not a contract.  `_mm512_cvtepi32_ps` is declared
 * in zmmintrin.h but lowers to `_mm512_cvt_roundepi32_ps`, which the KNC
 * runtime does not provide -> "undefined reference" at LINK time.
 * `_mm512_mullo_epi16` is likewise declared and likewise unlinkable.
 *
 * So every candidate below sits behind its own `TRY_<name>` switch.  The build
 * harness compiles this file once per switch (-DTRY_<name>) and records whether
 * the LINK succeeds; that is the ground truth.  Compiling with all the switches
 * that passed at once produces a single runnable program that also reports the
 * SEMANTICS of each primitive (operand order, lane-locality, byte vs dword
 * granularity), because several KNC names behave unlike their AVX-512 relatives.
 *
 * Primitives exercised: byte/dword permutes and shuffles (LUT candidates),
 * the unpack tree, byte->dword widening (cvtepu8_epi32), variable shifts,
 * masked loads and the int->float conversion candidates.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

static volatile int32_t vsink;
static volatile float   fsink;

/* ---------- test vectors ---------- */
/* one distinct value per 128-bit lane so lane-locality is visible */
static __m512i base_i(void) {
    int32_t t[16] __attribute__((aligned(64))) =
        {10,11,12,13, 20,21,22,23, 30,31,32,33, 40,41,42,43};
    return _mm512_load_epi32((const void *)t);
}
/* lane-local index pattern: 0,1,2,3 repeated in every lane */
static __m512i laneidx(void) {
    int32_t t[16] __attribute__((aligned(64))) =
        {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3};
    return _mm512_load_epi32((const void *)t);
}
/* 64 distinct bytes 0..63 */
static __m512i bytebase(void) {
    uint8_t t[64] __attribute__((aligned(64)));
    for (int i = 0; i < 64; ++i) t[i] = (uint8_t)i;
    return _mm512_load_epi32((const void *)t);
}
/* float LUT {-1,0,1,2} in the first 128-bit lane, replicated in all four */
static __m512 lut_f(void) {
    float t[16] __attribute__((aligned(64))) =
        {-1.f,0.f,1.f,2.f, -1.f,0.f,1.f,2.f, -1.f,0.f,1.f,2.f, -1.f,0.f,1.f,2.f};
    return _mm512_load_ps(t);
}

/* 64-byte aligned scratch so the load/store probes use real pointers.
 * NOTE: __m512i is a *union* on k1om, so `(const void *)some__m512i` is not a
 * legal cast -- the earlier load probes failed to compile for that reason, not
 * because the intrinsic was missing. */
static uint8_t g_buf[192] __attribute__((aligned(64)));

static void pv(const char *tag, __m512i v) {
    int32_t t[16] __attribute__((aligned(64)));
    _mm512_store_ps((float *)t, _mm512_castsi512_ps(v));
    printf("  %-24s dwords: %d %d %d %d | %d %d %d %d | %d %d %d %d | %d %d %d %d\n",
           tag, t[0],t[1],t[2],t[3], t[4],t[5],t[6],t[7],
           t[8],t[9],t[10],t[11], t[12],t[13],t[14],t[15]);
}
static void pb(const char *tag, __m512i v) {
    uint8_t t[64] __attribute__((aligned(64)));
    _mm512_store_ps((float *)t, _mm512_castsi512_ps(v));
    printf("  %-24s bytes : ", tag);
    for (int i = 0; i < 16; ++i) printf("%02x ", t[i]);
    printf("\n");
}
static void pfl(const char *tag, __m512 v) {
    float t[16] __attribute__((aligned(64)));
    _mm512_store_ps(t, v);
    printf("  %-24s floats: %.0f %.0f %.0f %.0f | %.0f %.0f %.0f %.0f\n",
           tag, t[0],t[1],t[2],t[3], t[4],t[5],t[6],t[7]);
}

int main(void) {
    const __m512i A   = base_i();
    const __m512i IDX = laneidx();
    const __m512i B   = bytebase();
    const __m512  LF  = lut_f();

    printf("=== KNC (k1om) IMCI link + semantics probe ===\n");
    printf("  base_i() = 10 11 12 13 | 20 21 22 23 | 30 31 32 33 | 40 41 42 43\n");
    for (int i = 0; i < 192; ++i) g_buf[i] = (uint8_t)i;

    /* ---------------- baseline: known good ---------------- */
    {
        __m512i r = _mm512_add_epi32(A, _mm512_set1_epi32(1));
        vsink = _mm512_reduce_add_epi32(r);
        printf("  [baseline] _mm512_add_epi32 / set1_epi32 / reduce_add_epi32 OK\n");
        r = _mm512_and_si512(A, _mm512_set1_epi32(3));
        vsink = _mm512_reduce_add_epi32(r);
        printf("  [baseline] _mm512_and_si512 OK\n");
        __m512 fr = _mm512_mul_ps(LF, _mm512_set1_ps(2.0f));
        fsink = _mm512_reduce_add_ps(fr);
        printf("  [baseline] _mm512_mul_ps / set1_ps / reduce_add_ps OK\n");
    }

    /* ---------------- LUT / permute candidates ---------------- */
#ifdef TRY_permutevar_epi32
    {   /* data first, index second (KNC order, unlike AVX-512's permutexvar) */
        __m512i r = _mm512_permutevar_epi32(A, IDX);
        pv("permutevar_epi32", r);
        vsink = _mm512_reduce_add_epi32(r);
        /* the actual LUT use: index = code 0..3 -> float */
        __m512i c = _mm512_and_si512(B, _mm512_set1_epi32(0x03030303));
        __m512i f = _mm512_permutevar_epi32(_mm512_castps_si512(LF), c);
        pfl("permutevar_epi32(LUT)", _mm512_castsi512_ps(f));
    }
#endif

#ifdef TRY_permutexvar_epi32
    {   /* AVX-512 operand order: index first */
        __m512i r = _mm512_permutexvar_epi32(IDX, A);
        pv("permutexvar_epi32", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_mask_permutexvar_epi32
    {
        __m512i r = _mm512_maskz_permutexvar_epi32(0x0F0F, IDX, A);
        pv("maskz_permutexvar", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_mask_permutevar_epi32
    {
        __m512i r = _mm512_mask_permutevar_epi32(_mm512_set1_epi32(-7), (__mmask16)0x00FF, A, IDX);
        pv("mask_permutevar", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_shuffle_epi8
    {   /* vpshufb: per-128-bit-lane byte LUT, ctrl = 0x0f - i reverses each lane */
        __m512i ctrl = _mm512_sub_epi8(_mm512_set1_epi8(15), B);
        __m512i r = _mm512_shuffle_epi8(B, ctrl);
        pb("shuffle_epi8(rev)", r);
        vsink = _mm512_reduce_add_epi32(r);
        /* the real use: 16-byte LUT lane = 4 floats {-1,0,1,2}, ctrl = 4*q+b */
        __m512i c = _mm512_and_si512(B, _mm512_set1_epi32(0x03030303));
        __m512i s = _mm512_slli_epi16(c, 2);
        __m512i pat = _mm512_set1_epi32(0x03020100);
        __m512i r2 = _mm512_shuffle_epi8(_mm512_castps_si512(LF), _mm512_or_si512(s, pat));
        pb("shuffle_epi8(LUT)", r2);
        vsink = _mm512_reduce_add_epi32(r2);
    }
#endif

#ifdef TRY_swizzle_epi32
    {
        __m512i r = _mm512_swizzle_epi32(A, _MM_SWIZ_REG_BADC);
        pv("swizzle_epi32(BADC)", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_shuffle_epi32
    {
        __m512i r = _mm512_shuffle_epi32(A, _MM_PERM_BADC);
        pv("shuffle_epi32(BADC)", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_permute4f128_epi32
    {
        __m512i r = _mm512_permute4f128_epi32(A, _MM_PERM_BADC);
        pv("permute4f128(BADC)", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_shuffle_i32x4
    {
        __m512i r = _mm512_shuffle_i32x4(A, A, _MM_PERM_CDAB);
        pv("shuffle_i32x4(CDAB)", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif

#ifdef TRY_mask_permutevar_ps
    {
        __m512 r = _mm512_mask_permutevar_ps(LF, (__mmask16)0xFFFF,
                                             _mm512_castsi512_ps(IDX));
        pfl("mask_permutevar_ps", r);
        fsink = _mm512_reduce_add_ps(r);
    }
#endif

#ifdef TRY_permutevar_ps
    {
        __m512 r = _mm512_permutevar_ps(LF, _mm512_castsi512_ps(IDX));
        pfl("permutevar_ps", r);
        fsink = _mm512_reduce_add_ps(r);
    }
#endif

    /* ---------------- unpack tree ---------------- */
#ifdef TRY_unpacklo_epi8
    { __m512i r = _mm512_unpacklo_epi8(B, _mm512_setzero_si512()); pb("unpacklo_epi8", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpackhi_epi8
    { __m512i r = _mm512_unpackhi_epi8(B, _mm512_setzero_si512()); pb("unpackhi_epi8", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpacklo_epi16
    { __m512i r = _mm512_unpacklo_epi16(B, _mm512_setzero_si512()); pb("unpacklo_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpackhi_epi16
    { __m512i r = _mm512_unpackhi_epi16(B, _mm512_setzero_si512()); pb("unpackhi_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpacklo_epi32
    { __m512i r = _mm512_unpacklo_epi32(A, _mm512_setzero_si512()); pv("unpacklo_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpackhi_epi32
    { __m512i r = _mm512_unpackhi_epi32(A, _mm512_setzero_si512()); pv("unpackhi_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpacklo_epi64
    { __m512i r = _mm512_unpacklo_epi64(A, _mm512_setzero_si512()); pv("unpacklo_epi64", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_unpackhi_epi64
    { __m512i r = _mm512_unpackhi_epi64(A, _mm512_setzero_si512()); pv("unpackhi_epi64", r); vsink=_mm512_reduce_add_epi32(r); }
#endif

    /* ---------------- widening / narrowing ---------------- */
#ifdef TRY_cvtepu8_epi32
    {   /* vpmovzxbd: 16 bytes -> 16 dwords.  The key primitive for this task. */
        __m128i s = _mm_loadu_si128((const __m128i *)(const void *)"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f");
        __m512i r = _mm512_cvtepu8_epi32(s);
        pv("cvtepu8_epi32", r);
        vsink = _mm512_reduce_add_epi32(r);
    }
#endif
#ifdef TRY_cvtepu16_epi32
    { __m256i s = _mm256_loadu_si256((const __m256i *)(const void *)B); __m512i r = _mm512_cvtepu16_epi32(s); pv("cvtepu16_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_cvtepu8_epi16
    { __m128i s = _mm_loadu_si128((const __m128i *)(const void *)B); __m512i r = _mm512_cvtepu8_epi16(s); pb("cvtepu8_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_cvtepi8_epi32
    { __m128i s = _mm_loadu_si128((const __m128i *)(const void *)B); __m512i r = _mm512_cvtepi8_epi32(s); pv("cvtepi8_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_cvtepi16_epi32
    { __m256i s = _mm256_loadu_si256((const __m256i *)(const void *)B); __m512i r = _mm512_cvtepi16_epi32(s); pv("cvtepi16_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif

    /* ---------------- shifts / logic ---------------- */
#ifdef TRY_slli_epi16
    { __m512i r = _mm512_slli_epi16(B, 2); pb("slli_epi16(2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_srli_epi16
    { __m512i r = _mm512_srli_epi16(B, 2); pb("srli_epi16(2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_slli_epi32
    { __m512i r = _mm512_slli_epi32(A, 2); pv("slli_epi32(2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_srli_epi32
    { __m512i r = _mm512_srli_epi32(A, 2); pv("srli_epi32(2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_srai_epi32
    { __m512i r = _mm512_srai_epi32(_mm512_set1_epi32(-8), 1); pv("srai_epi32(-8,1)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_srlv_epi32
    { __m512i r = _mm512_srlv_epi32(A, IDX); pv("srlv_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_sllv_epi32
    { __m512i r = _mm512_sllv_epi32(A, IDX); pv("sllv_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_or_si512
    { __m512i r = _mm512_or_si512(A, IDX); pv("or_si512", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_xor_si512
    { __m512i r = _mm512_xor_si512(A, IDX); pv("xor_si512", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_andnot_si512
    { __m512i r = _mm512_andnot_si512(A, _mm512_set1_epi32(-1)); pv("andnot_si512", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_sub_epi32
    { __m512i r = _mm512_sub_epi32(A, _mm512_set1_epi32(11)); pv("sub_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mullo_epi32
    { __m512i r = _mm512_mullo_epi32(A, _mm512_set1_epi32(3)); pv("mullo_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mullo_epi16
    { __m512i r = _mm512_mullo_epi16(A, _mm512_set1_epi16(3)); pv("mullo_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_madd_epi16
    { __m512i r = _mm512_madd_epi16(A, _mm512_set1_epi16(3)); pv("madd_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_maddubs_epi16
    { __m512i r = _mm512_maddubs_epi16(_mm512_and_si512(B, _mm512_set1_epi8(1)), _mm512_set1_epi8(2)); pb("maddubs_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_abs_epi32
    { __m512i r = _mm512_abs_epi32(_mm512_sub_epi32(_mm512_setzero_si512(), A)); pv("abs_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif

    /* ---------------- blend / mask ---------------- */
#ifdef TRY_mask_blend_epi32
    { __m512i r = _mm512_mask_blend_epi32((__mmask16)0x00FF, A, _mm512_set1_epi32(-1)); pv("mask_blend_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_mov_epi32
    { __m512i r = _mm512_mask_mov_epi32(A, (__mmask16)0x000F, _mm512_set1_epi32(-1)); pv("mask_mov_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_set1_epi32
    { __m512i r = _mm512_mask_set1_epi32(A, (__mmask16)0x000F, -1); pv("mask_set1_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_maskz_set1_epi32
    { __m512i r = _mm512_maskz_set1_epi32((__mmask16)0x000F, -1); pv("maskz_set1_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_movm_epi32
    { __m512i r = _mm512_movm_epi32((__mmask16)0xAAAA); pv("movm_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif

    /* ---------------- loads / stores ---------------- */
#ifdef TRY_maskz_load_epi32
    { __m512i r = _mm512_maskz_load_epi32((__mmask16)0x00FF, (const void *)g_buf); pv("maskz_load_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_load_epi32
    { __m512i r = _mm512_mask_load_epi32(_mm512_set1_epi32(-1), (__mmask16)0x00FF, (const void *)g_buf); pv("mask_load_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_store_epi32
    { int32_t t[16] __attribute__((aligned(64))); _mm512_store_epi32((void *)t, A); printf("  %-24s dwords: %d %d %d %d\n", "store_epi32", t[0],t[1],t[2],t[3]); vsink = t[0]; }
#endif

    /* ---------------- int -> float ---------------- */
#ifdef TRY_cvtepi32_ps
    {
        __m512 f = _mm512_cvtepi32_ps(_mm512_set1_epi32(3));
        pfl("cvtepi32_ps", f);
        fsink = _mm512_reduce_add_ps(f);
    }
#endif
#ifdef TRY_cvt_roundepi32_ps
    {
        __m512 f = _mm512_cvt_roundepi32_ps(_mm512_set1_epi32(3), _MM_FROUND_TO_NEAREST_INT);
        pfl("cvt_roundepi32_ps", f);
        fsink = _mm512_reduce_add_ps(f);
    }
#endif

    /* ---------------- round 2: loads, stores, misc ---------------- */
#ifdef TRY_loadu_si512
    { __m512i r = _mm512_loadu_si512((const void *)(g_buf + 1)); pb("loadu_si512(+1)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_maskz_loadu_epi32
    { __m512i r = _mm512_maskz_loadu_epi32((__mmask16)0x00FF, (const void *)(g_buf + 1)); pv("maskz_loadu_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_loadu_epi32
    { __m512i r = _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), (__mmask16)0x00FF, (const void *)(g_buf + 1)); pv("mask_loadu_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_storeu_si512
    { _mm512_storeu_si512((void *)(g_buf + 65), A); printf("  %-24s bytes : %02x %02x %02x %02x\n", "storeu_si512(+1)", g_buf[65], g_buf[66], g_buf[67], g_buf[68]); vsink = g_buf[65]; }
#endif
#ifdef TRY_extload_epi32
    { __m512i r = _mm512_extload_epi32((const void *)g_buf, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE); pv("extload_epi32(NONE)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_extload_epi32_u8
    { __m512i r = _mm512_extload_epi32((const void *)g_buf, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE); pv("extload_epi32(U8)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_extload_epi32_u8
    { __m512i r = _mm512_mask_extload_epi32(A, (__mmask16)0x00FF, (const void *)g_buf, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE); pv("mask_extload(U8)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_extloadunpacklo_epi32
    { __m512i r = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), (const void *)(g_buf + 2), _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE); pv("extloadunpacklo(+2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_extloadunpackhi_epi32
    { __m512i r = _mm512_extloadunpackhi_epi32(_mm512_setzero_si512(), (const void *)(g_buf + 2), _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE); pv("extloadunpackhi(+2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_extloadunpacklo_u8
    { __m512i r = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), (const void *)(g_buf + 2), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE); pv("extloadunpacklo U8(+2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_extloadunpackhi_u8
    { __m512i r = _mm512_extloadunpackhi_epi32(_mm512_setzero_si512(), (const void *)(g_buf + 2), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE); pv("extloadunpackhi U8(+2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_extloadunpacklo_epi32
    { __m512i r = _mm512_mask_extloadunpacklo_epi32(_mm512_setzero_si512(), (__mmask16)0xFFFF, (const void *)(g_buf + 2), _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE); pv("mask_extloadunpacklo", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_mask_extloadunpacklo_u8
    { __m512i r = _mm512_mask_extloadunpacklo_epi32(_mm512_setzero_si512(), (__mmask16)0xFFFF, (const void *)(g_buf + 2), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE); pv("mask_extloadunpacklo U8", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_gather_u8
    {   /* gather 16 bytes with UINT8 upconversion directly from +2 */
        __m512i gi = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        __m512i r = _mm512_i32extgather_epi32(gi, (const void *)(g_buf + 2), _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE);
        pv("gather U8(+2) scale1", r); vsink=_mm512_reduce_add_epi32(r);
    }
#endif
#ifdef TRY_gather_dwords
    {   /* gather the 8 packed dwords of a block payload with scale 4 */
        __m512i gi = _mm512_set_epi32(7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0);
        __m512i r = _mm512_i32extgather_epi32(gi, (const void *)(g_buf + 2), _MM_UPCONV_EPI32_NONE, 4, _MM_HINT_NONE);
        pv("gather dwords(+2)", r); vsink=_mm512_reduce_add_epi32(r);
    }
#endif
#ifdef TRY_gather_dwords_scale1
    {   /* gather 8 packed dwords using byte offsets and scale 1 */
        __m512i gi = _mm512_set_epi32(28,24,20,16,12,8,4,0, 28,24,20,16,12,8,4,0);
        __m512i r = _mm512_i32extgather_epi32(gi, (const void *)(g_buf + 2), _MM_UPCONV_EPI32_NONE, 1, _MM_HINT_NONE);
        pv("gather dw scale1(+2)", r); vsink=_mm512_reduce_add_epi32(r);
    }
#endif
#ifdef TRY_stream_load_si512
    { __m512i r = _mm512_stream_load_si512((const void *)g_buf); pv("stream_load_si512", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_loadu_ps
    { __m512 r = _mm512_loadu_ps((const void *)(g_buf + 4)); pfl("loadu_ps(+4)", r); fsink=_mm512_reduce_add_ps(r); }
#endif
#ifdef TRY_storeu_ps
    { _mm512_storeu_ps((void *)(g_buf + 68), LF); printf("  %-24s floats: %.0f %.0f\n", "storeu_ps(+4)", ((float *)(g_buf + 68))[0], ((float *)(g_buf + 68))[1]); fsink = ((float *)(g_buf + 68))[0]; }
#endif
#ifdef TRY_mask_storeu_epi32
    { _mm512_mask_storeu_epi32((void *)(g_buf + 4), (__mmask16)0x00FF, A); vsink = *(int32_t *)(g_buf + 4); printf("  %-24s dword0: %d\n", "mask_storeu_epi32", vsink); }
#endif
#ifdef TRY_alignr_epi32
    { __m512i r = _mm512_alignr_epi32(A, _mm512_set1_epi32(-1), 1); pv("alignr_epi32(1)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_permutex2var_epi32
    { __m512i r = _mm512_permutex2var_epi32(A, IDX, _mm512_set1_epi32(-1)); pv("permutex2var_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_maskz_permutexvar_epi32
    { __m512i r = _mm512_maskz_permutexvar_epi32((__mmask16)0x0F0F, IDX, A); pv("maskz_permutexvar", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_maskz_permutevar_epi32
    { __m512i r = _mm512_maskz_permutevar_epi32((__mmask16)0x0F0F, A, IDX); pv("maskz_permutevar", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_permutevar_ps
    { __m512 r = _mm512_permutevar_ps(LF, _mm512_castsi512_ps(IDX)); pfl("permutevar_ps", r); fsink=_mm512_reduce_add_ps(r); }
#endif
#ifdef TRY_mask_permutevar_ps
    { __m512 r = _mm512_mask_permutevar_ps(LF, (__mmask16)0xFFFF, LF, _mm512_castsi512_ps(IDX)); pfl("mask_permutevar_ps", r); fsink=_mm512_reduce_add_ps(r); }
#endif
#ifdef TRY_maskz_permutevar_ps
    { __m512 r = _mm512_maskz_permutevar_ps((__mmask16)0x000F, LF, _mm512_castsi512_ps(IDX)); pfl("maskz_permutevar_ps", r); fsink=_mm512_reduce_add_ps(r); }
#endif
#ifdef TRY_shuffle_i32x4
    { __m512i r = _mm512_shuffle_i32x4(A, A, _MM_PERM_CDAB); pv("shuffle_i32x4(CDAB)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_sub_epi8
    { __m512i r = _mm512_sub_epi8(B, _mm512_set1_epi8(1)); pb("sub_epi8", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_set1_epi8
    { __m512i r = _mm512_set1_epi8(7); pb("set1_epi8", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_set1_epi16
    { __m512i r = _mm512_set1_epi16(7); pb("set1_epi16", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_setzero_si512
    { __m512i r = _mm512_setzero_si512(); pv("setzero_si512", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_set_epi32
    { __m512i r = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0); pb("set_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_set_epi64
    { __m512i r = _mm512_set_epi64(7,6,5,4,3,2,1,0); pb("set_epi64", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_load_epi32
    { __m512i r = _mm512_load_epi32((const void *)g_buf); pb("load_epi32(aligned)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_maskz_mov_epi32
    { __m512i r = _mm512_maskz_mov_epi32((__mmask16)0x00FF, A); pv("maskz_mov_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_max_epi32
    { __m512i r = _mm512_max_epi32(A, _mm512_set1_epi32(21)); pv("max_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_min_epi32
    { __m512i r = _mm512_min_epi32(A, _mm512_set1_epi32(21)); pv("min_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_cmpeq_epi32_mask
    { __mmask16 k = _mm512_cmpeq_epi32_mask(A, _mm512_set1_epi32(21)); printf("  %-24s mask  : 0x%04X\n", "cmpeq_epi32_mask", (unsigned)k); vsink = (int32_t)k; }
#endif
#ifdef TRY_cmplt_epi32_mask
    { __mmask16 k = _mm512_cmplt_epi32_mask(A, _mm512_set1_epi32(21)); printf("  %-24s mask  : 0x%04X\n", "cmplt_epi32_mask", (unsigned)k); vsink = (int32_t)k; }
#endif
#ifdef TRY_i32extgather_epi32
    { __m512i r = _mm512_i32extgather_epi32(IDX, (const void *)g_buf, _MM_UPCONV_EPI32_NONE, 4, _MM_HINT_NONE); pv("i32extgather_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_slli_epi64
    { __m512i r = _mm512_slli_epi64(A, 2); pv("slli_epi64(2)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_add_epi32
    { __m512i r = _mm512_add_epi32(A, A); pv("add_epi32", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_bslli_epi128
    { __m512i r = _mm512_bslli_epi128(B, 1); pb("bslli_epi128(1)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif
#ifdef TRY_bsrli_epi128
    { __m512i r = _mm512_bsrli_epi128(B, 1); pb("bsrli_epi128(1)", r); vsink=_mm512_reduce_add_epi32(r); }
#endif

    printf("  (done)\n");
    return (int)(vsink + fsink);
}
