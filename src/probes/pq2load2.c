/* pq2load2.c -- semantics of the k1om unaligned load paths, one per process
 * invocation (a fault in one test must not hide the others).
 *   ./pq2load2 <n>
 * g_buf[i] = i so the expected bytes are obvious.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

static uint8_t g_buf[256] __attribute__((aligned(64)));

static void pv16(const char *tag, __m512i v) {
    int32_t t[16] __attribute__((aligned(64)));
    _mm512_store_epi32((void *)t, v);
    printf("  %-26s %08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x\n",
           tag, t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],
           t[8],t[9],t[10],t[11],t[12],t[13],t[14],t[15]);
}
static void pb16(const char *tag, __m512i v) {
    uint8_t t[64] __attribute__((aligned(64)));
    _mm512_store_epi32((void *)t, v);
    printf("  %-26s bytes:", tag);
    for (int i = 0; i < 20; ++i) printf(" %02x", t[i]);
    printf(" ...\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 0; i < 256; ++i) g_buf[i] = (uint8_t)i;
    const int t = (argc > 1) ? atoi(argv[1]) : 0;
    printf("pq2load2 test %d  (g_buf[i] = i)\n", t);

    switch (t) {
    case 1:
        pv16("extload U8 +0", _mm512_extload_epi32((const void *)g_buf,
             _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE)); break;
    case 2:
        pv16("extload U8 +4", _mm512_extload_epi32((const void *)(g_buf+4),
             _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE)); break;
    case 3:
        pv16("extload U8 +8", _mm512_extload_epi32((const void *)(g_buf+8),
             _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE)); break;
    case 4:
        pv16("extload U8 +16", _mm512_extload_epi32((const void *)(g_buf+16),
             _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE)); break;
    case 5:
        pv16("extload NONE +2", _mm512_extload_epi32((const void *)(g_buf+2),
             _MM_UPCONV_EPI32_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE)); break;
    case 6:
        pv16("unpacklo NONE +2", _mm512_extloadunpacklo_epi32(
             _mm512_setzero_si512(), (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE)); break;
    case 7:
        pv16("unpackhi NONE +2", _mm512_extloadunpackhi_epi32(
             _mm512_setzero_si512(), (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE)); break;
    case 8:
        pv16("unpacklo U8 +2", _mm512_extloadunpacklo_epi32(
             _mm512_setzero_si512(), (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)); break;
    case 9:
        pv16("unpackhi U8 +2", _mm512_extloadunpackhi_epi32(
             _mm512_setzero_si512(), (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)); break;
    case 10: {
        __m512i v = _mm512_setzero_si512();
        v = _mm512_extloadunpacklo_epi32(v, (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(g_buf+66),
             _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        pb16("lo(+2)+hi(+66) NONE", v);
        break; }
    case 11: {
        __m512i v = _mm512_setzero_si512();
        v = _mm512_extloadunpacklo_epi32(v, (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(g_buf+18),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv16("lo(+2)+hi(+18) U8", v);
        break; }
    case 12: {
        __m512i gi = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        pv16("gather U8 scale1 +2", _mm512_i32extgather_epi32(gi,
             (const void *)(g_buf+2), _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE));
        break; }
    case 13: {
        __m512i gi = _mm512_set_epi32(7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0);
        pv16("gather dw scale4 +2", _mm512_i32extgather_epi32(gi,
             (const void *)(g_buf+2), _MM_UPCONV_EPI32_NONE, 4, _MM_HINT_NONE));
        break; }
    case 14: {
        __m512i gi = _mm512_set_epi32(28,24,20,16,12,8,4,0,28,24,20,16,12,8,4,0);
        pv16("gather dw scale1 +2", _mm512_i32extgather_epi32(gi,
             (const void *)(g_buf+2), _MM_UPCONV_EPI32_NONE, 1, _MM_HINT_NONE));
        break; }
    case 15:
        pv16("mask_unpacklo U8 +2", _mm512_mask_extloadunpacklo_epi32(
             _mm512_setzero_si512(), (__mmask16)0x00FF, (const void *)(g_buf+2),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)); break;
    case 16: {
        /* does lo alone already give the full 32-byte payload? */
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)(g_buf+2), _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        pb16("lo NONE +2 bytes", v);
        break; }
    default:
        printf("  (no such test)\n");
    }
    printf("  done\n");
    return 0;
}
