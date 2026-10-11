/* probe_b.c -- what does _mm512_reduce_add_ps actually sum on k1om?
 *
 * probe_a showed the mirrored kernel returns exactly HALF for a single
 * 128-weight block (mirror = -64 where the per-lane dump proved every one of
 * the 16 lanes of every accumulator holds -1.0, so the sum must be -128).
 * That points at the reduction, not the accumulation.
 *
 * Here the 16 lanes hold distinct values 1000+k (k = 0..15, true sum = 16120),
 * so a partial reduction identifies exactly which lanes are missing.
 *
 * Header contract (zmmintrin.h:2876-2883): "_mm512_reduce_add_ps returns
 * float32 value calculated as v1[0] + v1[1] + ... + v1[15]".
 *
 * NOTE: _mm512_extractf32x4_ps / _mm512_shuffle_f32x4 COMPILE but fail to LINK
 * on this target (undefined reference), so lane extraction here goes through
 * _mm512_store_ps + 128-bit loads, which is known to work.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

int main(void) {
    float v[16] __attribute__((aligned(64)));
    for (int k = 0; k < 16; ++k) v[k] = 1000.0f + (float)k;

    double true_sum = 0.0;
    for (int k = 0; k < 16; ++k) true_sum += (double)v[k];

    printf("=== probe_b: _mm512_reduce_add_ps lane coverage (k1om) ===\n");
    printf("  v[k] = 1000+k for k=0..15;  true sum = %.1f\n\n", true_sum);

    __m512 vec = _mm512_load_ps(v);

    /* --- route 1: the intrinsic used by the kernel --- */
    const float r_intr = _mm512_reduce_add_ps(vec);
    printf("  1) _mm512_reduce_add_ps(vec)              = %10.1f  %s\n",
           (double)r_intr, ((double)r_intr == true_sum) ? "OK" : "PARTIAL");

    /* --- route 2: store, then four 128-bit loads summed by hand --- */
    {
        float s[16] __attribute__((aligned(64)));
        _mm512_store_ps(s, vec);
        printf("  2) store_ps + four 128-bit quarter sums:\n");
        double tot = 0.0;
        for (int g = 0; g < 4; ++g) {
            __m128 q = _mm_load_ps(s + g * 4);
            float qq[4] __attribute__((aligned(16)));
            _mm_store_ps(qq, q);
            double qs = (double)qq[0] + qq[1] + qq[2] + qq[3];
            tot += qs;
            printf("       quarter %d = %7.1f %7.1f %7.1f %7.1f  (sum %8.1f)\n",
                   g, qq[0], qq[1], qq[2], qq[3], qs);
        }
        printf("       q0+q1+q2+q3                          = %10.1f\n", tot);
        printf("       q0+q1 only                           = %10.1f\n",
               (double)s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7]);
        printf("       q2+q3 only                           = %10.1f\n",
               (double)s[8] + s[9] + s[10] + s[11] + s[12] + s[13] + s[14] + s[15]);
    }

    /* --- route 3: scalar partial sums within each 128-bit quarter --- */
    {
        float s[16] __attribute__((aligned(64)));
        _mm512_store_ps(s, vec);
        double tot = 0.0;
        for (int g = 0; g < 4; ++g) {
            float p = 0.0f;
            for (int k = 0; k < 4; ++k) p += s[g * 4 + k];
            tot += (double)p;
        }
        printf("  3) scalar sums within each 128-bit quarter = %10.1f\n", tot);
    }

    /* --- route 4: fully scalar --- */
    {
        float s[16] __attribute__((aligned(64)));
        _mm512_store_ps(s, vec);
        double tot = 0.0;
        for (int k = 0; k < 16; ++k) tot += (double)s[k];
        printf("  4) store_ps + scalar sum                  = %10.1f\n", tot);
    }

    /* --- route 5: low half vs high half of the stored image --- */
    {
        float s[16] __attribute__((aligned(64)));
        _mm512_store_ps(s, vec);
        double lo8 = 0, hi8 = 0;
        for (int k = 0; k < 8; ++k)  lo8 += (double)s[k];
        for (int k = 8; k < 16; ++k) hi8 += (double)s[k];
        printf("  5) mem[0..7] = %.1f   mem[8..15] = %.1f   all = %.1f\n",
               lo8, hi8, lo8 + hi8);
        printf("       intrinsic/lo8 = %.6f   intrinsic/hi8 = %.6f\n",
               (double)r_intr / lo8, (double)r_intr / hi8);
        printf("       (1.0 on one of them => the intrinsic saw only that half)\n");
    }

    /* --- subset search: which lane subset sums to the intrinsic answer? --- */
    printf("\n  subset search: which subsets sum to the intrinsic answer?\n");
    {
        int found = 0;
        for (int start = 0; start <= 12; ++start) {
            double s = 0.0;
            for (int k = start; k < start + 4; ++k) s += (double)v[k];
            if (fabs(s - (double)r_intr) < 0.5) {
                printf("       MATCH: lanes %d..%d (sum %.1f)\n", start, start + 3, s);
                found = 1;
            }
        }
        for (int start = 0; start <= 8; start += 4) {
            double s = 0.0;
            for (int k = start; k < start + 8; ++k) s += (double)v[k];
            if (fabs(s - (double)r_intr) < 0.5) {
                printf("       MATCH: lanes %d..%d (sum %.1f)\n", start, start + 7, s);
                found = 1;
            }
        }
        if (!found) {
            /* is it the plain halving of the true sum? */
            printf("       no 4/8-lane subset matches; intrinsic = %.1f, "
                   "true/2 = %.1f\n", (double)r_intr, true_sum / 2.0);
        }
    }

    printf("\n  true/intrinsic = %.6f  (2.0 => exactly half)\n",
           true_sum / (double)r_intr);

    /* --- same question for all-identical lanes, the kernel's regime --- */
    {
        float o[16] __attribute__((aligned(64)));
        for (int k = 0; k < 16; ++k) o[k] = -1.0f;
        printf("\n  identical lanes all -1.0: reduce = %.1f (true -16.0)\n",
               (double)_mm512_reduce_add_ps(_mm512_load_ps(o)));
        for (int k = 0; k < 16; ++k) o[k] = -40.0f;
        printf("  identical lanes all -40.0: reduce = %.1f (true -640.0)\n",
               (double)_mm512_reduce_add_ps(_mm512_load_ps(o)));
    }
    return 0;
}
