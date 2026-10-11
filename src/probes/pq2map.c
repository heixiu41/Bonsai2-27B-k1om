/* Empirically determine the lane -> byte mapping of PQ2_IMCI_LOAD16.
 *
 * Why this exists: a decoder variant that avoids the broadcast permute
 * segfaulted, and the blocker was that the lane->byte correspondence of the
 * proven load primitive had never been measured directly -- it was only ever
 * inferred from the fact that pq2_imci_idx + SHIFT together give the right
 * answer. Optimising against an unmeasured assumption is how the
 * "extloadunpacklo silently zero-fills" bug nearly shipped.
 *
 * Method notes (learned the hard way -- earlier versions of this probe
 * segfaulted twice):
 *   - every vector store destination must be 64-byte aligned;
 *   - LOAD16(p) reads up to p+81, and _mm512_extload_epi32 in UINT8 mode wants
 *     16-byte alignment, so keep p 16-byte aligned and leave >= 96 bytes of
 *     buffer after p.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#include "pq2_0.h"

/* buf[k] = k, so a lane's byte value identifies its source offset */
static uint8_t buf[512] __attribute__((aligned(64)));

static void show(const char *tag, __m512i v) {
    int l[16] __attribute__((aligned(64)));
    _mm512_store_epi32(l, v);
    printf("%-26s :", tag);
    for (int i = 0; i < 16; ++i) printf(" %3u", (unsigned)(l[i] & 0xFF));
    printf("\n");
}

int main(void) {
    for (int k = 0; k < 512; ++k) buf[k] = (uint8_t)k;

    printf("=== PQ2_IMCI_LOAD16(p = buf+off): lane values ===\n");
    printf("=== want: lane i == buf[off+i] (identity) ===\n\n");
    const int offs[] = { 0, 16, 32, 48, 64, 96, 144, 208, 256, 320 };
    for (unsigned o = 0; o < sizeof(offs)/sizeof(offs[0]); ++o) {
        const int off = offs[o];
        char tag[48];
        snprintf(tag, sizeof tag, "LOAD16(buf+%d)", off);
        show(tag, PQ2_IMCI_LOAD16(buf + off));
        printf("%-26s :", "  expected (off+i)");
        for (int i = 0; i < 16; ++i) printf(" %3u", (unsigned)buf[off + i]);
        printf("\n\n");
    }

    printf("=== _mm512_extload_epi32(p, UINT8): dword packing ===\n\n");
    for (unsigned o = 0; o < 4; ++o) {
        const int off = offs[o];
        char tag[48];
        snprintf(tag, sizeof tag, "extload(buf+%d)", off);
        show(tag, _mm512_extload_epi32((const void *)(buf + off),
                                       _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE, 0));
    }
    printf("\nIf lane i packs bytes 4i..4i+3 little-endian, then at off=0:\n");
    printf("  lane 0 should be 0x03020100 = %d\n", (int)0x03020100u);
    {
        int l[16] __attribute__((aligned(64)));
        _mm512_store_epi32(l, _mm512_extload_epi32((const void *)buf,
                              _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE, 0));
        printf("  measured lane 0 = 0x%08X   lane 1 = 0x%08X\n",
               (unsigned)l[0], (unsigned)l[1]);
    }
    return 0;
}
