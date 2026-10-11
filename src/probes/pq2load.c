/* pq2load.c -- minimal isolation of the crash in pq2imci:
 * which part of the extload path faults?  Each step prints before it runs and
 * stdout is unbuffered, so the last line printed identifies the faulting step.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

static volatile int32_t sink;
static volatile float   fsink;

static void pv(const char *tag, __m512i v) {
    int32_t t[16] __attribute__((aligned(64)));
    _mm512_store_epi32((void *)t, v);
    printf("  %-30s %d %d %d %d | %d %d %d %d\n", tag,
           t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);
}

int main(void) {
    static uint8_t buf[256] __attribute__((aligned(64)));
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 0; i < 256; ++i) buf[i] = (uint8_t)i;
    printf("pq2load: extload alignment isolation\n");

    printf("A) aligned extload UINT8 from buf+0\n");
    pv("aligned +0", _mm512_extload_epi32((const void *)buf,
        _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));

    printf("B) aligned extload UINT8 from buf+16\n");
    pv("aligned +16", _mm512_extload_epi32((const void *)(buf + 16),
        _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));

    printf("C) UNALIGNED extload UINT8 from buf+2\n");
    pv("unaligned +2", _mm512_extload_epi32((const void *)(buf + 2),
        _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));

    printf("D) UNALIGNED extload UINT8 from buf+34\n");
    pv("unaligned +34", _mm512_extload_epi32((const void *)(buf + 34),
        _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));

    printf("E) UNALIGNED extload NONE from buf+2\n");
    pv("unaligned NONE +2", _mm512_extload_epi32((const void *)(buf + 2),
        _MM_UPCONV_EPI32_NONE, _MM_BROADCAST32_NONE, _MM_HINT_NONE));

    printf("F) permutevar index=(+2 row) data=lut\n");
    {
        __m512i lut = _mm512_set_epi32((int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u,
                                       (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u,
                                       (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u,
                                       (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u);
        __m512i idx = _mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0);
        __m512i r = _mm512_permutevar_epi32(idx, lut);
        fsink = _mm512_reduce_add_ps(_mm512_castsi512_ps(r));
        pv("permutevar(lut)", r);
    }

    printf("G) store_ps into a 64B-aligned heap buffer\n");
    {
        float *o = (float *)aligned_alloc(64, 512);
        __m512 v = _mm512_set1_ps(1.5f);
        _mm512_store_ps(o, v);
        printf("  o[0]=%.2f\n", o[0]);
        free(o);
    }

    printf("H) stack array with aligned attribute + store_ps\n");
    {
        float a[128] __attribute__((aligned(64)));
        _mm512_store_ps(a, _mm512_set1_ps(2.5f));
        printf("  a[0]=%.2f  (ptr%%64=%d)\n", a[0], (int)((uintptr_t)a & 63));
    }

    printf("all steps completed\n");
    return 0;
}
