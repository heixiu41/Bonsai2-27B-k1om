/* repack_probe.c -- step 1 of the codes-only repack: prove the lane mapping.
 *
 * The repack replaces each PQ2_IMCI_LOAD16 (vpxord + vloadunpackld +
 * vloadunpackhd, arbitrary byte offset) with a single aligned
 * vmovdqa32 {uint8} once the block stride is 32 instead of 34.  The load
 * halves are then 16-byte aligned, which is what the upconverting load needs.
 *
 * The risk is the LANE ORDER: the "unpack" forms are documented to compose
 * into "dword i = p[i]", but a plain vmovdqa32 {uint8} could swizzle instead.
 * This measures, on the card, for every 16-byte-aligned offset in a
 * 64-byte-aligned buffer:
 *   - lane-by-lane equality of extload_epi32(UINT8) vs PQ2_IMCI_LOAD16
 *   - the lane map of each form on a ramp pattern (so a mismatch is readable)
 *   - the same for the packed-pair geometry (offsets 0/16 of a 32-byte block)
 *
 * Build: icc -mmic -O2 -std=c11 repack_probe.c -o repack_probe
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#define PQ2_IMCI_LOAD16(p) \
    _mm512_extloadunpackhi_epi32( \
        _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), \
            (const void *)(p), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE), \
        (const void *)((const uint8_t *)(p) + 64), \
        _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)

static inline __m512i aligned_ld(const void *p) {
    return _mm512_extload_epi32(p, _MM_UPCONV_EPI32_UINT8,
                                _MM_BROADCAST32_NONE, _MM_HINT_NONE);
}

/* 64-byte aligned buffer with slack so the loadunpack over-read stays inside */
static uint8_t buf[512] __attribute__((aligned(64)));
static int32_t la[16] __attribute__((aligned(64)));
static int32_t lb[16] __attribute__((aligned(64)));

int main(void) {
    for (int i = 0; i < (int)sizeof buf; ++i) buf[i] = (uint8_t)(i * 7 + 3);

    printf("=== lane map on a ramp (expect dword i = p[i]) ===\n");
    printf("%6s %8s %8s %8s\n", "i", "p[i]", "LOAD16", "aligned");
    _mm512_store_si512((void *)la, PQ2_IMCI_LOAD16(buf));
    _mm512_store_si512((void *)lb, aligned_ld(buf));
    for (int i = 0; i < 16; ++i)
        printf("%6d %8u %8d %8d%s\n", i, (unsigned)buf[i], la[i], lb[i],
               la[i] == lb[i] ? "" : "   <-- DIFFERENT");
    int bad16 = 0;
    for (int i = 0; i < 16; ++i)
        if (la[i] != lb[i]) ++bad16;
    printf("  mismatching lanes at offset 0: %d/16\n\n", bad16);

    printf("=== every 16-byte-aligned offset in a 64-byte-aligned buffer ===\n");
    int total = 0, cases = 0;
    for (int off = 0; off + 16 <= 256; off += 16) {
        _mm512_store_si512((void *)la, PQ2_IMCI_LOAD16(buf + off));
        _mm512_store_si512((void *)lb, aligned_ld(buf + off));
        int bad = 0;
        for (int i = 0; i < 16; ++i)
            if (la[i] != lb[i]) ++bad;
        ++cases;
        total += bad;
        if (bad)
            printf("  offset %3d: %2d/16 lanes differ  (p[0..3]=%u,%u,%u,%u  "
                   "lo[0..3]=%d,%d,%d,%d  al[0..3]=%d,%d,%d,%d)\n",
                   off, bad, buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
                   la[0], la[1], la[2], la[3], lb[0], lb[1], lb[2], lb[3]);
    }
    printf("  %d offsets checked, %d lane mismatches total\n\n", cases, total);

    printf("=== packed geometry: block stride 32, halves at 32b and 32b+16 ===\n");
    int blkbad = 0;
    for (int b = 0; b < 6; ++b) {
        const uint8_t *blk = buf + 32 * b;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *q = blk + 16 * half;
            _mm512_store_si512((void *)la, PQ2_IMCI_LOAD16(q));
            _mm512_store_si512((void *)lb, aligned_ld(q));
            int bad = 0;
            for (int i = 0; i < 16; ++i)
                if (la[i] != lb[i]) ++bad;
            if (bad) {
                ++blkbad;
                printf("  block %d half %d (off %d): %d/16 differ\n",
                       b, half, (int)(q - buf), bad);
            }
        }
    }
    printf("  packed offsets with mismatches: %d/12\n", blkbad);

    printf("\nVERDICT: %s\n",
           (total == 0 && blkbad == 0)
               ? "aligned extload_epi32(UINT8) == PQ2_IMCI_LOAD16 bit-for-bit"
               : "LANE MAPPINGS DIFFER -- the repack needs a compensating permute");
    return (total == 0 && blkbad == 0) ? 0 : 1;
}
