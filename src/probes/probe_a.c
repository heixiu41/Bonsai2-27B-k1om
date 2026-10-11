/* probe_a.c -- minimal, hand-checkable reproduction of the PQ2_0 IMCI
 * "wrong answer" defect.  Own file; pq2_0.h untouched.
 *
 * Mirrors pq2_0_matvec_knc's loop body verbatim (4 x 16-wide accumulators over
 * 128-weight blocks, then _mm512_reduce_add_ps) but in ONE row (d==1), so the
 * matrix has exactly one row's worth of bytes and there is no row-stride /
 * heap-corruption question at all.
 *
 * Cases are chosen so the exact answer is an integer computable by hand:
 *   n = 128  -> 1 block.  codes 0 (=-1), scale 1.0  ->  sum = -128
 *   n = 5120 -> 40 blocks, same weights           ->  sum = -5120
 * For every case it prints the reference, the mirrored kernel, and the four
 * accumulator vectors element by element, plus wf[0..3] and wf[64..67].
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "pq2_0.h"

#define FP16_ONE 0x3C00

static void *al(size_t bytes) {
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

static void fill_uniform(uint8_t *w, int nblk, int code, uint16_t h) {
    for (int b = 0; b < nblk; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = (uint8_t)(h & 0xFF);
        blk[1] = (uint8_t)(h >> 8);
        memset(blk + 2, code * 0x55, 32);
    }
}

/* ---- verbatim mirror of the kernel body, d == 1, with dumps ---- */
static double mirror_kernel(const float *x, const uint8_t *w, int n, int dump) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;
    float wf[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    int iters = 0;

    for (int j = 0; j < nn; j += QK_PQ2_0) {
        ++iters;
        pq2_0_dequant_block(w + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
        if (dump && (j == 0 || j == QK_PQ2_0 || j == nn - QK_PQ2_0)) {
            printf("      iter %2d (j=%5d): wf[0..3]=%.1f %.1f %.1f %.1f  "
                   "wf[64..67]=%.1f %.1f %.1f %.1f\n",
                   iters, j, wf[0], wf[1], wf[2], wf[3],
                   wf[64], wf[65], wf[66], wf[67]);
        }
        sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      sum0);
        sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), sum1);
        sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), sum3);
    }
    if (dump) {
        float t0[16] __attribute__((aligned(64)));
        float t1[16] __attribute__((aligned(64)));
        float t2[16] __attribute__((aligned(64)));
        float t3[16] __attribute__((aligned(64)));
        _mm512_store_ps(t0, sum0); _mm512_store_ps(t1, sum1);
        _mm512_store_ps(t2, sum2); _mm512_store_ps(t3, sum3);
        printf("      iterations = %d (expect %d)\n", iters, nn / QK_PQ2_0);
        for (int k = 0; k < 16; ++k)
            printf("      lane %2d: s0=%8.1f s1=%8.1f s2=%8.1f s3=%8.1f\n",
                   k, t0[k], t1[k], t2[k], t3[k]);
    }
    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return (double)_mm512_reduce_add_ps(sum);
}

static int run(const char *name, int n, int code, double expect, int dump) {
    const int nblk = n / QK_PQ2_0;
    uint8_t *w = (uint8_t *)al((size_t)nblk * PQ2_0_BLOCK_BYTES);
    float   *x = (float *)al((size_t)n * sizeof(float));
    float    yref = 0.0f;
    if (!w || !x) return 1;

    fill_uniform(w, nblk, code, FP16_ONE);
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    /* scalar reference in the template (compiled on the card on purpose) */
    pq2_0_matvec_ref(&yref, x, w, n, 1);

    const double m = mirror_kernel(x, w, n, dump);

    printf("  %-28s n=%5d  expect=%10.1f  ref=%10.1f  mirror=%10.1f  %s\n",
           name, n, expect, (double)yref, m,
           (fabs((double)yref - expect) > 1e-3) ? "REF-ALSO-WRONG"
                                                : (fabs(m - expect) < 1e-3 ? "OK" : "WRONG"));
    free(w); free(x);
    return 0;
}

int main(void) {
    printf("=== probe_a: minimal single-row IMCI kernel reproduction ===\n");
    printf("    all weights = -1.0 (fp16 1.0 x code 0), x = all ones\n\n");

    printf("  case 1: one 128-weight block\n");
    run("1 block, all -1", 128, 0, -128.0, 1);

    printf("\n  case 2: 40 blocks (5120 weights) -- where the earlier tests\n");
    printf("          reported exactly half (-2560 instead of -5120)\n");
    run("40 blocks, all -1", 5120, 0, -5120.0, 1);

    printf("\n  case 3: other uniform patterns, single row\n");
    run("40 blocks, all +1", 5120, 2, +5120.0, 0);
    run("40 blocks, all +2", 5120, 3, +10240.0, 0);
    run("40 blocks, all  0", 5120, 1, 0.0, 0);

    printf("\n  case 4: shapes\n");
    run("2 blocks",   256, 0, -256.0, 0);
    run("3 blocks",   384, 0, -384.0, 0);
    run("4 blocks",   512, 0, -512.0, 0);
    run("8 blocks",  1024, 0, -1024.0, 0);
    run("20 blocks", 2560, 0, -2560.0, 0);
    run("21 blocks", 2688, 0, -2688.0, 0);
    return 0;
}
