/* Second-round link probe: the intrinsics the first probe either missed or
 * reported inconsistently.
 *
 * The first probe (intr_probe.c) covered ~87 names but its report contained
 * contradictions -- it listed _mm512_slli_epi16/srli_epi16 among BOTH the
 * linkable and the undefined-reference sets. Since the whole lesson of this
 * project is "the linker is the ground truth, not the header", each candidate
 * here is probed in isolation so a single result cannot be misattributed.
 *
 * Focus: instructions that could plausibly replace a vpermd in the PQ2_0
 * decoder, i.e. cheap cross-lane byte/word movement.
 *
 * Build with:  icc -mmic -O2 -std=c11 -D__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES pq2probe2.c
 * Each #ifdef'd block that fails to link is reported by the linker; comment out
 * the failures and re-link until it builds, then read which ones survived.
 */
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

static uint8_t buf[256] __attribute__((aligned(64)));
static int lv[16] __attribute__((aligned(64)));

static void keep(__m512i v) { _mm512_store_epi32(lv, v); }
static void keepf(__m512 v) { _mm512_store_ps((float *)lv, v); }

int main(void) {
    for (int i = 0; i < 256; ++i) buf[i] = (uint8_t)i;
    const __m512i a = _mm512_load_epi32(buf);
    const __m512i b = _mm512_load_epi32(buf + 64);
    const __m512i m16 = _mm512_set1_epi16(3);
    const __m512i m8  = _mm512_set1_epi8(3);

    printf("=== second-round link probe ===\n");
    printf("(a program that links proves the listed intrinsics exist in the "
           "runtime library)\n\n");

    /* ---- byte / word shifts ---- */
#ifdef TRY_BSLLI
    keep(_mm512_bslli_epi128(a, 2));      printf("  bslli_epi128        OK\n");
#endif
#ifdef TRY_BSRLI
    keep(_mm512_bsrli_epi128(a, 2));      printf("  bsrli_epi128        OK\n");
#endif
#ifdef TRY_SLLI16
    keep(_mm512_slli_epi16(a, 2));        printf("  slli_epi16          OK\n");
#endif
#ifdef TRY_SRLI16
    keep(_mm512_srli_epi16(a, 2));        printf("  srli_epi16          OK\n");
#endif
#ifdef TRY_SRAI16
    keep(_mm512_srai_epi16(a, 2));        printf("  srai_epi16          OK\n");
#endif

    /* ---- byte movement ---- */
#ifdef TRY_SHUFFLE8
    keep(_mm512_shuffle_epi8(a, b));      printf("  shuffle_epi8        OK\n");
#endif
#ifdef TRY_ALIGNR8
    keep(_mm512_alignr_epi8(a, b, 3));    printf("  alignr_epi8         OK\n");
#endif
#ifdef TRY_ALIGNR32
    keep(_mm512_alignr_epi32(a, b, 3));   printf("  alignr_epi32        OK\n");
#endif
#ifdef TRY_ALIGNR64
    keep(_mm512_alignr_epi64(a, b, 3));   printf("  alignr_epi64        OK\n");
#endif
#ifdef TRY_SHUFFLE_I32X4
    keep(_mm512_shuffle_i32x4(a, b, 0x1B)); printf("  shuffle_i32x4     OK\n");
#endif
#ifdef TRY_PERM4F128
    keep(_mm512_permute4f128_epi32(a, 0x1B)); printf("  permute4f128_epi32 OK\n");
#endif
#ifdef TRY_PERMXVAR32
    keep(_mm512_permutexvar_epi32(a, b)); printf("  permutexvar_epi32   OK\n");
#endif
#ifdef TRY_PERMXVAR8
    keep(_mm512_permutexvar_epi8(a, b));  printf("  permutexvar_epi8    OK\n");
#endif
#ifdef TRY_PERMX2VAR32
    keep(_mm512_permutex2var_epi32(a, b, a)); printf("  permutex2var_epi32 OK\n");
#endif

    /* ---- broadcast ---- */
#ifdef TRY_BCASTB
    keep(_mm512_broadcastb_epi8(a));      printf("  broadcastb_epi8     OK\n");
#endif
#ifdef TRY_BCASTW
    keep(_mm512_broadcastw_epi16(a));     printf("  broadcastw_epi16    OK\n");
#endif
#ifdef TRY_BCASTD
    keep(_mm512_broadcastd_epi32(a));     printf("  broadcastd_epi32    OK\n");
#endif
#ifdef TRY_SWIZZLE
    keep(_mm512_swizzle_epi32(a, _MM_SWIZ_REG_BADC)); printf("  swizzle_epi32 OK\n");
#endif
#ifdef TRY_SHUFFLE32
    keep(_mm512_shuffle_epi32(a, _MM_PERM_BADC)); printf("  shuffle_epi32     OK\n");
#endif

    /* ---- integer multiply / mul-add ---- */
#ifdef TRY_MADDUBS
    keep(_mm512_maddubs_epi16(a, b));     printf("  maddubs_epi16       OK\n");
#endif
#ifdef TRY_MADD16
    keep(_mm512_madd_epi16(a, b));        printf("  madd_epi16          OK\n");
#endif
#ifdef TRY_MULLO16
    keep(_mm512_mullo_epi16(a, b));       printf("  mullo_epi16         OK\n");
#endif
#ifdef TRY_MULHI16
    keep(_mm512_mulhi_epi16(a, b));       printf("  mulhi_epi16         OK\n");
#endif
#ifdef TRY_MINMAX8
    keep(_mm512_max_epu8(a, b));          printf("  max_epu8            OK\n");
#endif
#ifdef TRY_ADD8
    keep(_mm512_add_epi8(a, b));          printf("  add_epi8            OK\n");
#endif
#ifdef TRY_SUB8
    keep(_mm512_sub_epi8(a, b));          printf("  sub_epi8            OK\n");
#endif
#ifdef TRY_AND8
    keep(_mm512_and_si512(a, m8));        printf("  and_si512           OK\n");
#endif

    /* ---- gather / scatter ---- */
#ifdef TRY_I32GATHER
    keep(_mm512_i32gather_epi32(a, (const void *)buf, 1)); printf("  i32gather_epi32 OK\n");
#endif
#ifdef TRY_I32EXTGATHER
    keep(_mm512_i32extgather_epi32(a, (const void *)buf, _MM_UPCONV_EPI32_NONE, 1, 0));
    printf("  i32extgather_epi32 OK\n");
#endif

    /* ---- cvt ---- */
#ifdef TRY_CVTEPS
    keepf(_mm512_cvtepi32_ps(a));         printf("  cvtepi32_ps         OK\n");
#endif

    (void)m16;
    printf("\ndone. any candidate NOT printed above is absent from the runtime.\n");
    printf("(build with the TRY_* macros you want to test; a missing symbol is a "
           "link error, which is the real answer)\n");
    return 0;
}
