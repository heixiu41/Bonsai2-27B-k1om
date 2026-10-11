/* pq2ld1.c -- can ONE vloadunpackld replace the two-instruction LOAD16 pair?
 *
 * PQ2_IMCI_LOAD16(p) = extloadunpacklo(zero, p, UINT8) then
 *                      extloadunpackhi(that, p+64, UINT8)
 * i.e. TWO instructions, and the p+64 access is what forces the documented
 * "leave >= 64 bytes of readable slack after every block" requirement.
 *
 * If the single extloadunpacklo already yields lane i = p[i] for every one of
 * the 64 source alignments, the pair is unnecessary: -2 instructions per 128
 * weights AND the over-read shrinks to "the rest of p's own 64-byte block".
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "pq2_0.h"

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    /* large arena, guard-free, but we only ever touch [off, off+96) */
    uint8_t *area = (uint8_t *)aligned_alloc(64, 4096);
    uint8_t pat[64];
    for (int i = 0; i < 64; ++i) pat[i] = (uint8_t)(i * 37 + 11);

    int bad_lo = 0, bad_pair = 0, bad_lo_dirty = 0, worst_off = -1;
    int tot = 0;
    printf("=== single vloadunpackld vs the two-instruction pair ===\n");
    printf("off  lo-wrong pair-wrong   (lo = one extloadunpacklo with zero dst)\n");
    for (int off = 0; off < 64; ++off) {
        memset(area, 0xAA, 4096);
        memcpy(area + off, pat, 64);
        int want[16], a[16], b[16], c[16];
        for (int i = 0; i < 16; ++i) want[i] = area[off + i];
        _mm512_store_epi32(a, _mm512_extloadunpacklo_epi32(
                _mm512_setzero_si512(), area + off, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE));
        _mm512_store_epi32(b, PQ2_IMCI_LOAD16(area + off));
        _mm512_store_epi32(c, _mm512_extloadunpacklo_epi32(
                _mm512_set1_epi32(0x7F7F7F7F), area + off, _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE));
        int wl = 0, wp = 0, wc = 0;
        for (int i = 0; i < 16; ++i) {
            if (a[i] != want[i]) ++wl;
            if (b[i] != want[i]) ++wp;
            if (c[i] != want[i]) ++wc;
        }
        bad_lo += wl; bad_pair += wp; bad_lo_dirty += wc;
        tot += 16;
        if (wl) { printf(" %2d      %2d        %2d\n", off, wl, wp); if (worst_off < 0) worst_off = off; }
    }
    printf("\n  total lanes checked per variant: %d\n", tot);
    printf("  one extloadunpacklo (zero dst)  : %d wrong lanes\n", bad_lo);
    printf("  PQ2_IMCI_LOAD16 (lo + hi pair)  : %d wrong lanes\n", bad_pair);
    printf("  one extloadunpacklo (dirty dst) : %d wrong lanes\n", bad_lo_dirty);
    printf("  first alignment where the single load fails: %d\n", worst_off);

    /* how far past p does each variant actually read?  probe with a guard page
     * is not available here, so instead: check that the single load's result is
     * independent of the bytes past the end of p's own 64-byte block. */
    {
        int dep = 0;
        for (int off = 0; off < 64; ++off) {
            uint8_t *big = (uint8_t *)aligned_alloc(64, 512);
            memset(big, 0x11, 512);
            memcpy(big + off, pat, 34);
            __m512i v1 = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), big + off,
                            _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE);
            int r1[16], r2[16];
            _mm512_store_epi32(r1, v1);
            /* poison everything past the 64-byte aligned block containing off */
            size_t blkend = (((size_t)(big + off) + 63) & ~(size_t)63) - (size_t)big;
            if (blkend < 512) memset(big + blkend, 0xEE, 512 - blkend);
            _mm512_store_epi32(r2, _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), big + off,
                            _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE));
            for (int i = 0; i < 16; ++i) if (r1[i] != r2[i]) ++dep;
            free(big);
        }
        printf("  single-load lanes that change when everything past p's own\n");
        printf("  64-byte block is poisoned: %d (0 => it never reads past that block)\n", dep);
    }
    return 0;
}
