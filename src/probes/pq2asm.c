/* pq2asm.c -- one tiny function per exotic/interesting intrinsic, compiled with
 * -mmic -S so the generated assembly reveals the REAL k1om mnemonic behind each
 * header declaration. Host-side only; never run. */
#include <immintrin.h>

/* --- 2^23 / table-lookup-elimination candidates ------------------------- */
__m512i f_fmadd233_epi32(__m512i a, __m512i b) { return _mm512_fmadd233_epi32(a, b); }
__m512  f_fmadd233_ps(__m512 a, __m512 b)      { return _mm512_fmadd233_ps(a, b); }
__m512  f_exp2a23_ps(__m512 a)                 { return _mm512_exp2a23_round_ps(a, _MM_FROUND_CUR_DIRECTION); }
__m512  f_scale_ps(__m512 a, __m512i b)        { return _mm512_scale_ps(a, b); }
__m512  f_rsqrt23_ps(__m512 a)                 { return _mm512_rsqrt23_ps(a); }
__m512  f_maxabs_ps(__m512 a)                  { return _mm512_maxabs_ps(a, a); }

/* --- conversions -------------------------------------------------------- */
/* cvtph_ps: unsupported (movdqa) */
/* cvtps_ph dropped */
__m512  f_cvtepi32_ps(__m512i a)               { return _mm512_cvtepi32_ps(a); }
/* extload_ps f16 dropped */
__m512  f_extload_ps_u8(void const* p)         { return _mm512_extload_ps(p, _MM_UPCONV_PS_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE); }
/* cvtepi8_epi32: compile-rejected */
/* cvtepu8_epi32: compile-rejected */
/* cvtfxpnt dropped */

/* --- 8/16-bit element ops ----------------------------------------------- */
__m512i f_add_epi8(__m512i a, __m512i b)       { return _mm512_add_epi8(a, b); }
__m512i f_sub_epi8(__m512i a, __m512i b)       { return _mm512_sub_epi8(a, b); }
__m512i f_add_epi16(__m512i a, __m512i b)      { return _mm512_add_epi16(a, b); }
__m512i f_sub_epi16(__m512i a, __m512i b)      { return _mm512_sub_epi16(a, b); }
__m512i f_adds_epi8(__m512i a, __m512i b)      { return _mm512_adds_epi8(a, b); }
__m512i f_subs_epi8(__m512i a, __m512i b)      { return _mm512_subs_epi8(a, b); }
__m512i f_adds_epu8(__m512i a, __m512i b)      { return _mm512_adds_epu8(a, b); }
__m512i f_abs_epi8(__m512i a)                  { return _mm512_abs_epi8(a); }
__m512i f_abs_epi16(__m512i a)                 { return _mm512_abs_epi16(a); }
__m512i f_avg_epu8(__m512i a, __m512i b)       { return _mm512_avg_epu8(a, b); }
__m512i f_sad_epu8(__m512i a, __m512i b)       { return _mm512_sad_epu8(a, b); }
__m512i f_dbsad_epu8(__m512i a, __m512i b, int i) { return _mm512_dbsad_epu8(a, b, i); }
__m512i f_max_epu8(__m512i a, __m512i b)       { return _mm512_max_epu8(a, b); }
__m512i f_mulhrs_epi16(__m512i a, __m512i b)   { return _mm512_mulhrs_epi16(a, b); }
__m512i f_maddubs_epi16(__m512i a, __m512i b)  { return _mm512_maddubs_epi16(a, b); }
__m512i f_madd_epi16(__m512i a, __m512i b)     { return _mm512_madd_epi16(a, b); }
__m512i f_mullo_epi16(__m512i a, __m512i b)    { return _mm512_mullo_epi16(a, b); }
__m512i f_slli_epi16(__m512i a)                { return _mm512_slli_epi16(a, 3); }
__m512i f_srli_epi16(__m512i a)                { return _mm512_srli_epi16(a, 3); }
__m512i f_shuffle_epi8(__m512i a, __m512i b)   { return _mm512_shuffle_epi8(a, b); }
__m512i f_alignr_epi8(__m512i a, __m512i b)    { return _mm512_alignr_epi8(a, b, 3); }
__m512i f_unpacklo_epi8(__m512i a, __m512i b)  { return _mm512_unpacklo_epi8(a, b); }
__m512i f_unpacklo_epi16(__m512i a, __m512i b) { return _mm512_unpacklo_epi16(a, b); }
__m512i f_packs_epi16(__m512i a, __m512i b)    { return _mm512_packs_epi16(a, b); }
__m512i f_multishift(__m512i a, __m512i b)     { return _mm512_multishift_epi64_epi8(a, b); }
/* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: /* FAILED: __m512i f_broadcastb(__m128i a)                { return _mm512_broadcastb_epi8(a); } /* expect FAIL */ */ */ */ */ */ */ */ */ */ */ */ */ */ */ */ */ */ */ */ */
/* FAILED: __mmask64 f_cmpgt_epi8_mask(__m512i a, __m512i b) { return _mm512_cmpgt_epi8_mask(a, b); } */

/* --- permutes / shifts / logic ------------------------------------------ */
/* FAILED: __m512i f_permutevar(__m512i i, __m512i d)     { return _mm512_permutevar_epi32(i, d); } */
/* FAILED: __m512i f_permute4f128_i(__m512i d, int e)     { return _mm512_permute4f128_epi32(d, e); } */
/* FAILED: __m512  f_permute4f128_ps(__m512 d, int e)     { return _mm512_permute4f128_ps(d, e); } */
/* FAILED: __m512i f_swizzle_i(__m512i d, _MM_SWIZZLE_ENUM e) { return _mm512_swizzle_epi32(d, e); } */
/* FAILED: __m512  f_swizzle_ps(__m512 d, _MM_SWIZZLE_ENUM e) { return _mm512_swizzle_ps(d, e); } */
/* FAILED: __m512i f_shuffle_epi32(__m512i d, _MM_PERM_ENUM e) { return _mm512_shuffle_epi32(d, e); } */
/* FAILED: __m512  f_shuffle_ps(__m512 a, __m512 b, int i) { return _mm512_shuffle_ps(a, b, i); } */
/* FAILED: __m512i f_alignr_epi32(__m512i a, __m512i b)   { return _mm512_alignr_epi32(a, b, 3); } */
/* FAILED: __m512i f_alignr_epi64(__m512i a, __m512i b)   { return _mm512_alignr_epi64(a, b, 3); } */
/* FAILED: __m512i f_ternary(__m512i a, __m512i b, __m512i c) { return _mm512_ternarylogic_epi32(a, b, c, 0xCA); } */
/* FAILED: __m512i f_shuffle_i32x4(__m512i a, __m512i b, int i) { return _mm512_shuffle_i32x4(a, b, i); } */
/* FAILED: __m512i f_permutexvar(__m512i i, __m512i d)    { return _mm512_permutexvar_epi32(i, d); } */
/* FAILED: __m512i f_permutex2var(__m512i a, __m512i i, __m512i b) { return _mm512_permutex2var_epi32(a, i, b); } */
/* FAILED: __m512i f_permutexvar_epi16(__m512i i, __m512i d) { return _mm512_permutexvar_epi16(i, d); } */
/* FAILED: __m512i f_bslli(__m512i a)                     { return _mm512_bslli_epi128(a, 3); } */
/* FAILED: __m512i f_bsrli(__m512i a)                     { return _mm512_bsrli_epi128(a, 3); } */
/* FAILED: __m512i f_rolv_epi32(__m512i a, __m512i b)     { return _mm512_rolv_epi32(a, b); } */
/* FAILED: __m512i f_ror_epi32(__m512i a)                 { return _mm512_ror_epi32(a, 3); } */
__m512i f_sllv_epi32(__m512i a, __m512i b)     { return _mm512_sllv_epi32(a, b); }
__m512i f_srlv_epi32(__m512i a, __m512i b)     { return _mm512_srlv_epi32(a, b); }
__m512i f_sllv_epi16(__m512i a, __m512i b)     { return _mm512_sllv_epi16(a, b); }
__m512i f_mullo_epi32(__m512i a, __m512i b)    { return _mm512_mullo_epi32(a, b); }
__m512i f_mulhi_epi32(__m512i a, __m512i b)    { return _mm512_mulhi_epi32(a, b); }
/* popcnt absent */
__m512i f_lzcnt_epi32(__m512i a)               { return _mm512_lzcnt_epi32(a); }
__m512i f_conflict_epi32(__m512i a)            { return _mm512_conflict_epi32(a); }
__m512 f_addsets_ps(__m512 a, __m512 b, __mmask16* k) { return _mm512_addsets_ps(a, b, k); }
__m512i f_subsetb_epi32(__m512i a, __m512i b, __mmask16* k) { return _mm512_subsetb_epi32(a, b, k); }
__m512i f_addsetc_epi32(__m512i a, __m512i b, __mmask16* k) { return _mm512_addsetc_epi32(a, b, k); }
__m512  f_gmaxabs_ps(__m512 a, __m512 b)       { return _mm512_gmaxabs_ps(a, b); }
__m512  f_fixupnan_ps(__m512 a, __m512 b, __m512i c) { return _mm512_fixupnan_ps(a, b, c); }
__m512i f_gather_u16(__m512i off, void const *p) { return _mm512_i32extgather_epi32(off, p, _MM_UPCONV_EPI32_UINT16, 1, _MM_HINT_NONE); }
__m512  f_gather_ps(__m512i off, void const *p) { return _mm512_i32extgather_ps(off, p, _MM_UPCONV_PS_FLOAT16, 1, _MM_HINT_NONE); }
__m512i f_extload_u8(void const *p)            { return _mm512_extload_epi32(p, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE); }
__m512i f_extloadunpacklo_u8(void const *p, __m512i a) { return _mm512_extloadunpacklo_epi32(a, p, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE); }
