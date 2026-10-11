/* Settle two things the link probe just contradicted:
 *
 *   1. _mm512_cvtepi32_ps  -- reported EARLY in this project as unlinkable
 *      (`undefined reference to _mm512_cvt_roundepi32_ps`), but the
 *      per-macro link probe now says a program using it links. Either the
 *      earlier conclusion was wrong, or the call was optimised away. The only
 *      way to tell is to make the RESULT observable at runtime and compare it
 *      against the scalar conversion -- if the intrinsic were a no-op stub the
 *      values would not match.
 *
 *   2. _mm512_mullo_epi16 / slli_epi16 / srli_epi16 -- use them and verify the
 *      results, since several plausible decoder rewrites depend on them.
 *
 * Everything here is checked against scalar arithmetic, so a silently wrong
 * intrinsic cannot pass.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

static volatile int sink;
static int lv[16] __attribute__((aligned(64)));

int main(void) {
    printf("=== 1. _mm512_cvtepi32_ps: really present, really correct? ===\n\n");
    {
        /* lane i = i - 8, so we test negative, zero and positive */
        const __m512i v = _mm512_set_epi32(
            7,6,5,4, 3,2,1,0, -1,-2,-3,-4, -5,-6,-7,-8);
        const __m512 f = _mm512_cvtepi32_ps(v);
        float lf[16] __attribute__((aligned(64)));
        _mm512_store_ps(lf, f);
        int bad = 0;
        for (int i = 0; i < 16; ++i) {
            const int want_int = (i < 8) ? (i - 8) : (i - 8);
            const float want = (float)want_int;
            if (lf[i] != want) { printf("  lane %2d: got %g want %g\n", i, lf[i], want); ++bad; }
        }
        printf("  stored floats:");
        for (int i = 0; i < 16; ++i) printf(" %g", lf[i]);
        printf("\n  expected     : -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7\n");
        printf("  => %s\n\n", bad == 0 ? "CORRECT and PRESENT (the early 'unlinkable' "
                                        "conclusion was WRONG)"
                                      : "MISMATCH -> the intrinsic is a stub / broken");
        if (bad) printf("  BAD LANES = %d\n", bad);
    }

    printf("=== 2. word (16-bit) shifts: slli/srli_epi16 ===\n\n");
    {
        /* two 2-bit fields per dword: 0x0302 -> low word 0x0302, high word 0 */
        const __m512i v = _mm512_set1_epi32(0x00000302);
        int r[16] __attribute__((aligned(64)));

        _mm512_store_epi32(r, _mm512_srli_epi16(v, 2));
        printf("  srli_epi16(0x0302, 2)  = 0x%08X  (want 0x000000C0)\n",
               (unsigned)r[0]);
        const int ok1 = ((unsigned)r[0] == 0x000000C0u);

        _mm512_store_epi32(r, _mm512_slli_epi16(v, 2));
        printf("  slli_epi16(0x0302, 2)  = 0x%08X  (want 0x00000C08)\n",
               (unsigned)r[0]);
        const int ok2 = ((unsigned)r[0] == 0x00000C08u);

        _mm512_store_epi32(r, _mm512_srai_epi16(v, 2));
        printf("  srai_epi16(0x0302, 2)  = 0x%08X  (want 0x000000C0)\n",
               (unsigned)r[0]);

        printf("  => 16-bit shifts behave as documented: %s\n\n",
               (ok1 && ok2) ? "YES" : "NO");
    }

    printf("=== 3. _mm512_mullo_epi16 ===\n\n");
    {
        /* 16 independent 16-bit products per vector: lane pattern 2*3 = 6 */
        const __m512i a = _mm512_set1_epi32(0x00030002);   /* words: 2, 3 */
        const __m512i b = _mm512_set1_epi32(0x00050004);   /* words: 4, 5 */
        int r[16] __attribute__((aligned(64)));
        _mm512_store_epi32(r, _mm512_mullo_epi16(a, b));
        printf("  mullo_epi16(0x0003_0002, 0x0005_0004) = 0x%08X\n", (unsigned)r[0]);
        printf("  want low word 2*4=8=0x0008, high word 3*5=15=0x000F -> 0x000F0008\n");
        printf("  => %s\n\n", (unsigned)r[0] == 0x000F0008u ? "CORRECT" : "MISMATCH");
    }

    printf("=== 4. what this means for the decoder ===\n\n");
    printf("  If cvtepi32_ps really works, the integer-code path becomes viable:\n");
    printf("     k = (byte >> 2q) & 3      (integer)\n");
    printf("     f = cvtepi32_ps(k - 1)    (no 16-entry LUT permute needed!)\n");
    printf("     g = f * d\n");
    printf("  That would remove the LUT vpermd -- the more expensive of the two\n");
    printf("  permutes (~415 ps/weight of the ~765 ps total).\n");
    sink = lv[0];
    return 0;
}
