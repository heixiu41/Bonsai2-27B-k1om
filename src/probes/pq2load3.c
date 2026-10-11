/* pq2load3.c -- how do we get 16 arbitrary bytes when the window CROSSES a
 * 64-byte boundary?
 *
 * Established already (pq2load2.c): _mm512_extloadunpacklo_epi32(old, p,
 * UINT8, hint) delivers bytes [p .. end of the 64-byte aligned block
 * containing p] widened to dwords, and fills the rest with `old`.  With
 * old = 0 the bytes past the boundary come back as ZERO, which silently
 * corrupts a PQ2_0 block whenever qs&63 > 32.
 *
 * This program finds the primitive that does the crossing correctly.
 * base is 64-byte aligned and g_buf[i] = i, so the expected dwords are
 * obvious.  One test per process: a fault must not hide the other results.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

static uint8_t g_buf[512] __attribute__((aligned(64)));

static void pv(const char *tag, __m512i v) {
    int32_t t[16] __attribute__((aligned(64)));
    _mm512_store_epi32((void *)t, v);
    printf("  %-28s", tag);
    for (int i = 0; i < 16; ++i) printf(" %d", t[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 0; i < 512; ++i) g_buf[i] = (uint8_t)i;
    const int t = (argc > 1) ? atoi(argv[1]) : 0;
    /* base+52: 12 bytes left in the aligned block, so the 16-byte window at
     * qs = base+52 needs 4 bytes out of the NEXT aligned block. */
    const uint8_t *qs = g_buf + 52;
    printf("test %d (want bytes 52..67 = 52 53 54 ... 67)\n", t);

    switch (t) {
    case 1:
        pv("A lo U8 qs", _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE));
        break;
    case 2: {
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(g_buf + 64),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("lo(qs)+hi(base+64)", v);
        break; }
    case 3: {
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(qs + 64),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("lo(qs)+hi(qs+64)", v);
        break; }
    case 4: {
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(qs + 16),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("lo(qs)+hi(qs+16)", v);
        break; }
    case 5: {
        __m512i gi = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        pv("gather U8 s1 qs", _mm512_i32extgather_epi32(gi, (const void *)qs,
             _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE));
        break; }
    case 6: {
        __m512i gi = _mm512_set_epi32(31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16);
        pv("gather U8 s1 qs..+31", _mm512_i32extgather_epi32(gi, (const void *)qs,
             _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE));
        break; }
    case 7: {
        __m512i gi = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        pv("gather U8 s1 qs+16", _mm512_i32extgather_epi32(gi, (const void *)(qs+16),
             _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE));
        break; }
    case 8:
        pv("lo NONE qs (align?)", _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE));
        break;
    case 9: {
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(qs + 64),
             _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        pv("lo+hi NONE qs", v);
        break; }
    case 10: {
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)qs, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_mask_extloadunpacklo_epi32(v, (__mmask16)0xFF00,
             (const void *)(qs + 8), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("masked lo U8 +8", v);
        break; }
    case 11: {   /* do it in 8-byte halves: never crosses */
        __m512i v = _mm512_mask_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (__mmask16)0x00FF, (const void *)qs,
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_mask_extloadunpacklo_epi32(v, (__mmask16)0xFF00,
             (const void *)(qs + 8), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("8+8 masked lo", v);
        break; }
    case 12: {   /* same but the second half from qs+8 == crossing point */
        __m512i v = _mm512_mask_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (__mmask16)0x000F, (const void *)qs,
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_mask_extloadunpacklo_epi32(v, (__mmask16)0xFFF0,
             (const void *)(qs + 4), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("4+12 masked lo", v);
        break; }
    case 13: {   /* gather the two halves with the SAME index vector, base 0/16 */
        __m512i gi = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        __m512i a = _mm512_i32extgather_epi32(gi, (const void *)qs,
             _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE);
        __m512i b = _mm512_i32extgather_epi32(gi, (const void *)(qs + 16),
             _MM_UPCONV_EPI32_UINT8, 1, _MM_HINT_NONE);
        pv("gather a (qs)", a);
        pv("gather b (qs+16)", b);
        break; }
    /* ---- alignment requirements, one offset per run ---- */
    case 20: case 21: case 22: case 23: case 24: {
        /* NONE lo+hi at offset t-20 (0,1,2,3,4) : what alignment does it need? */
        const int off = t - 20;
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)(g_buf + off), _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(g_buf + off + 64),
             _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
        pv("NONE lo+hi off", v);
        printf("  offset %d (ptr%%4=%d): dword0 should be %08x\n",
               off, (int)((uintptr_t)(g_buf+off) & 3), *(const uint32_t *)(g_buf+off));
        break; }
    case 30: case 31: case 32: case 33: {
        /* UINT8 lo+hi at odd/high offsets: does crossing work at ANY offset? */
        static const int offs[4] = { 1, 3, 53, 63 };
        const int off = offs[t - 30];
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)(g_buf + off), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(g_buf + off + 64),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("U8 lo+hi off", v);
        printf("  offset %d: want %d %d %d ... %d\n", off,
               g_buf[off], g_buf[off+1], g_buf[off+2], g_buf[off+15]);
        break; }
    case 40: case 41: {
        /* UINT8 lo+hi at qs+16 (second half of a block payload) */
        const int off = (t == 40) ? 53 : 63;
        const uint8_t *p = g_buf + off + 16;
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
             (const void *)p, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        v = _mm512_extloadunpackhi_epi32(v, (const void *)(p + 64),
             _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
        pv("U8 lo+hi qs+16", v);
        printf("  qs+16 = %d: want %d %d ... %d\n", off + 16,
               g_buf[off+16], g_buf[off+17], g_buf[off+31]);
        break; }
    default:
        printf("  (no such test)\n");
    }
    printf("  done\n");
    return 0;
}
