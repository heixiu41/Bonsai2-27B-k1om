/* Isolate the decoder from the FMA: compute, for each output lane, the exact
 * 2-bit code that V1 (shipped) and V8 (vpmulld candidate) send into the LUT.
 *
 * Why: an end-to-end kernel comparison only says "the results differ". Printing
 * the index each version actually looks up pinpoints whether V8 selects the
 * wrong BYTE or the wrong FIELD, which is the difference between two very
 * different fixes.
 *
 * Semantics used here were measured on this card (src/bonsai/pq2sem.c):
 *   - permutevar_epi32(idx, data): index is the FIRST argument;
 *     result[i] = data[idx[i] & 15]
 *   - set_epi32(a15,...,a0): the LAST argument is lane 0
 *   - srlv is a per-lane right shift
 *   - the LUT selects by (index & 15), and LUT lane j = (j & 3) - 1
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#include "pq2_0.h"

static int lane_i(__m512i v, int i) {
    int l[16] __attribute__((aligned(64)));
    _mm512_store_epi32(l, v);
    return l[i & 15];
}

int main(void) {
    /* one block: scale = 1.0, code bytes = 0x00..0x1F (distinct per byte) */
    uint8_t blk[PQ2_0_BLOCK_BYTES] __attribute__((aligned(64)));
    memset(blk, 0, sizeof blk);
    blk[0] = 0x00; blk[1] = 0x3C;                    /* fp16 1.0 */
    for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(i & 0x3F);
    /* make the fields differ within each byte so a wrong field shows up:
     * byte b = b,  so fields are (b>>0)&3, (b>>2)&3, (b>>4)&3, (b>>6)&3 */
    const uint8_t *qs = blk + 2;

    const __m512i c0 = PQ2_IMCI_LOAD16(qs);
    const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
    const __m512i shf = PQ2_IMCI_SHIFT();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);

    printf("=== per-lane LUT index: V1 vs V8 ===\n");
    printf("(index = the 2-bit code; the LUT turns j into (j&3)-1)\n\n");
    printf("%4s %8s %8s %8s   %8s %8s %8s\n",
           "lane", "V1idx", "V8idx", "match", "V1byte", "V8byte", "V1fld/V8fld");

    int mismatches = 0;
    for (int i = 0; i < 128; ++i) {
        int v1idx, v8idx, v1byte, v8byte, v1fld, v8fld;

        /* ---- V1: broadcast byte B = 16*(i/16) + (i/4)%4, then field i%4 ---- */
        {
            const int m  = (i / 16) & 3;
            const __m512i c = (i < 64) ? c0 : c1;
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            /* physical lane carrying output lane i is: q + 4*m  where q=i%4,
             * and the LUT picks byte (lane&3) of v -- replicate that read */
            const int lane = (i % 4) + 4 * ((i / 16) & 3);
            v1idx  = lane_i(v, lane) & 15;
            v1idx &= 3;                       /* the LUT masks &15 then &3 */
            v1byte = (i < 64) ? (16 * (i / 16) + (i / 4) % 4)
                              : (16 + 16 * ((i - 64) / 16) + (i / 4) % 4);
            v1fld  = i % 4;
        }

        /* ---- V8: shift this lane's own dword by 2q + 8m, then replicate ---- */
        {
            const int m = (i / 16) & 3;
            const __m512i c = (i < 64) ? c0 : c1;
            const int shift = 2 * (i % 4) + 8 * m;
            const __m512i k = _mm512_srlv_epi32(c, _mm512_set1_epi32(shift));
            const __m512i t = _mm512_mullo_epi32(k, mulc);
            /* LUT reads byte (lane & 3) of the physical lane carrying lane i */
            const int lane = (i % 4) + 4 * ((i / 16) & 3);
            v8idx = (lane_i(t, lane) >> (8 * (lane & 3))) & 0xFF;
            v8idx &= 3;
            v8byte = (i < 64) ? (4 * ((i / 16)) + (i / 4) % 4)
                              : (16 + 4 * ((i - 64) / 16) + (i / 4) % 4);
            v8fld  = i % 4;
        }

        const int ok = (v1idx == v8idx);
        if (!ok) ++mismatches;
        if (i < 24 || !ok)
            printf("%4d %8d %8d %8s   %8d %8d  %d/%d\n",
                   i, v1idx, v8idx, ok ? "ok" : "DIFF",
                   v1byte, v8byte, v1fld, v8fld);
    }
    printf("\nlane-index mismatches = %d / 128\n", mismatches);
    printf("=> the decode arithmetic is %s\n",
           mismatches == 0 ? "IDENTICAL (so the bug is NOT in the index math)"
                           : "DIFFERENT (see the byte/field columns)");

    /* Cross-check against the scalar definition for the first few lanes */
    printf("\n=== scalar ground truth for the first 8 weights ===\n");
    for (int j = 0; j < 8; ++j) {
        const int q = (qs[j / 4] >> ((j % 4) * 2)) & 3;
        printf("  weight %d: byte qs[%d]=0x%02X field %d -> code %d -> value %d\n",
               j, j / 4, qs[j / 4], j % 4, q, q - 1);
    }
    return 0;
}
