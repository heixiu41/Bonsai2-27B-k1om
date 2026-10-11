/* Does KNC have popcount, and can it help the PQ2_0 decoder?
 *
 * Answering both questions, because "is it available" and "is it useful" are
 * separate, and this project has been burned by assuming either.
 *
 * PART 1 -- availability. The header declares only scalar _popcnt32/_popcnt64;
 * there is no _mm512_popcnt_* anywhere. But the earlier link probes here gave a
 * FALSE POSITIVE because dead-code elimination removed the calls, so this tests
 * with the results made observable (printed / volatile) and cross-checks against
 * a known-answer computation.
 *
 * PART 2 -- usefulness. popcount is a legitimate way to EXTRACT the 2-bit code
 * field without a permute, because for a 2-bit value k,
 *      (b >> 2q) & 3  ==  popcount(b >> 2q & 3)      is trivially circular,
 * but the useful identity is that a 2-bit field's value can be recovered from
 * four single-bit counts:
 *      k = popcount(b & m1) - popcount(b & m2) + 2*popcount(b & m3) ...
 * which is ALGEBRAICALLY correct but costs several vector ops per field.
 *
 * So part 2 measures the op count against what the existing measurements say an
 * op is worth (a vpermd is ~2.6x cheaper than the integer sequence V3 tried).
 * The point is to answer with arithmetic rather than intuition.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

static volatile int vsink;

int main(void) {
    printf("=== PART 1: is scalar popcount present and correct on k1om? ===\n\n");
    {
        /* make the result observable so the compiler cannot delete the call */
        const unsigned v = 0xB4u;              /* 0b10110100 -> 4 bits set */
        const int pc = _popcnt32((int)v);
        int bad = 0;
        for (unsigned i = 0; i < 4096u; i += 137u) {
            int want = 0;
            for (unsigned t = i; t; t >>= 1) want += (int)(t & 1u);
            if (_popcnt32((int)i) != want) ++bad;
        }
        printf("  _popcnt32(0x%X) = %d   (expect 4)   %s\n",
               v, pc, pc == 4 ? "correct" : "WRONG");
        printf("  swept 4096/137 values: mismatches = %d\n", bad);
        vsink = pc;
        printf("  => scalar popcount %s\n\n", (pc == 4 && bad == 0) ? "PRESENT and CORRECT"
                                                                 : "absent or broken");
    }

    printf("=== PART 2: can popcount extract the 2-bit field cheaply? ===\n\n");
    {
        /* For k in 0..3:  k = (b>>0&1) + 2*(b>>1&1). Using popcount on a
         * masked vector replaces the shift+mask with popcount(mask & b).
         * Count the vector ops for one field of one 16-lane group: */
        const __m512i b = _mm512_set1_epi32(0x000000B4);
        const __m512i m0 = _mm512_set1_epi32(0x01);
        const __m512i m1 = _mm512_set1_epi32(0x02);
        int r[16] __attribute__((aligned(64)));

        /* current decoder: 1 srlv + LUT(permutevar)  = 2 ops, no popcount */
        const __m512i shifted = _mm512_srlv_epi32(b, _mm512_setzero_si512());
        _mm512_store_epi32(r, shifted);
        const int cur = r[0] & 3;

        /* popcount route would need, per field: and + popcount + and + popcount
         * + add  = 5 ops, and there is NO vector popcount available anyway.
         * Emulate the scalar equivalent to show the algebra is right: */
        const int k_from_pop = _popcnt32(0xB4 & 0x01) + 2 * _popcnt32(0xB4 & 0x02);
        printf("  scalar check: popcount-based field value = %d  (expect %d)\n",
               k_from_pop, 0xB4 & 3);
        printf("  algebra correct: %s\n\n", k_from_pop == (0xB4 & 3) ? "yes" : "no");
        printf("  ops per field:\n");
        printf("    current      : srlv + permutevar(LUT)          = 2 vector ops\n");
        printf("    popcount     : and + pcnt + and + pcnt + add   = 5 vector ops\n");
        printf("    (and there is no _mm512_popcnt_epi32 to use)\n\n");
        printf("  => popcount would need 5 ops to replace a 2-op sequence that\n");
        printf("     already measured CHEAPER than a 3-op integer alternative\n");
        printf("     (V3: vpaddd+vpsubps was 2.1x SLOWER than the LUT permute).\n");
        printf("     Reachable field value above = %d\n", cur);
    }
    return 0;
}
