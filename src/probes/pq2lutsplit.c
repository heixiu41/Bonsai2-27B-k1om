/* Can ONE permutevar serve four LUTs at once?
 *
 * The idea: `permutevar(idx, v)` returns dword v[idx & 15]. A 512-bit vector is
 * 16 dwords = 64 bytes. If the LUT's 16 floats were spread across the BYTES of
 * those dwords rather than packed one-per-dword, then a single permute whose
 * index used bits 4..5 to choose the byte might serve several fields at once --
 * which would cut the decoder's 8 LUT permutes per 128 weights to 2.
 *
 * But the measured semantics say the index is masked with & 15, i.e. only the
 * low 4 bits are consulted, and the result is a whole DWORD. If that is exactly
 * right, the idea cannot work, because there is no way for the index to select
 * a byte within a dword.
 *
 * This program settles it by construction rather than argument:
 *   - build a vector whose bytes 0..3 of dword j are four different values
 *   - sweep the index across all 256 values 0..255 and record what comes back
 *   - if the result only ever depends on idx & 15 and always returns a full
 *     dword, the "byte-select" idea is disproven.
 *
 * Every result is printed, so nothing can be optimised away.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

int main(void) {
    printf("=== can permutevar select a BYTE, or only a whole dword? ===\n\n");

    /* dword j = 0x04030201 * (j+1) so its four bytes are distinct and the
     * dword as a whole is unique per j. */
    int d[16] __attribute__((aligned(64)));
    for (int j = 0; j < 16; ++j) d[j] = (int)(0x04030201u * (unsigned)(j + 1));
    const __m512i v = _mm512_load_epi32(d);

    printf("source dwords (hex):\n");
    for (int j = 0; j < 16; ++j) printf("  [%2d] 0x%08X\n", j, (unsigned)d[j]);
    printf("\n");

    /* Sweep the index. If only idx & 15 matters, then idx and idx+16 give the
     * same answer, and every answer is one of the source dwords. */
    int mismatch_pair = 0, non_dword = 0;
    printf("idx  result       is-a-source-dword?  same as idx&15?\n");
    for (int idx = 0; idx < 40; ++idx) {
        int r[16] __attribute__((aligned(64)));
        _mm512_store_epi32(r, _mm512_permutevar_epi32(_mm512_set1_epi32(idx), v));
        const unsigned got = (unsigned)r[0] & 0xFFFFFFFFu;
        int found = -1;
        for (int j = 0; j < 16; ++j) if ((unsigned)d[j] == got) { found = j; break; }
        if (found < 0) ++non_dword;
        const int expect = idx & 15;
        const int same = (found == expect);
        if (!same) ++mismatch_pair;
        if (idx < 20 || idx >= 32)
            printf("%3d  0x%08X  %-18s  %s\n", idx, got,
                   found < 0 ? "NO (partial byte!)" : "yes",
                   same ? "yes" : "NO");
    }
    printf("\n  values that are NOT one of the source dwords: %d\n", non_dword);
    printf("  cases where result != source[idx & 15]     : %d\n", mismatch_pair);
    printf("\n  => permutevar returns %s\n",
           non_dword == 0 && mismatch_pair == 0
             ? "a WHOLE DWORD selected by (idx & 15).\n     The 'spread the LUT across "
               "bytes' idea is DISPROVEN -- the index\n     has no bits left to choose a "
               "byte, and the result is a full dword."
             : "something byte-granular, so the idea may be viable -- investigate");

    printf("\n=== corollary: how many dwords of LUT would a multi-table scheme need? ===\n");
    printf("  A field's LUT has 4 distinct entries, but permutevar needs one entry\n");
    printf("  per DWORD and selects with 4 index bits, so a table occupies whole\n");
    printf("  dwords. Four separate tables in one vector would need 4 x 16 = 64\n");
    printf("  dwords = 4 vectors, and the index would have to select among 64\n");
    printf("  entries -- impossible with 4 index bits.\n");
    printf("  => one permute can serve exactly ONE 16-entry LUT.\n");
    return 0;
}
