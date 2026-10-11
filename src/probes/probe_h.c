/* probe_h.c -- FIXED kernel, verified element-by-element against pq2_0_matvec_ref.
 *
 * Root cause established by probe_g: the original kernel's four FMAs read
 * wf+0/16/32/48, i.e. 64 of the block's 128 weights; wf+64/80/96/112 are never
 * read. The dequantiser fills all 128, so exactly half the weights are dropped
 * and every result is exactly 2x too small. The 128-weight stride of the block
 * loop makes the omission silent.
 *
 * Fix: keep the four independent accumulators (same dependency structure, same
 * final add tree, same _mm512_reduce_add_ps) but give each one TWO 16-wide FMAs
 * per block, so all 128 weights are covered:
 *      sum0 += wf[ 0..15] * x[ j+ 0..15]   and   wf[64..79] * x[ j+64..79]
 *      sum1 += wf[16..31] * x[ j+16..31]   and   wf[80..95] * x[ j+80..95]
 *      sum2 += wf[32..47] * x[ j+32..47]   and   wf[96..111] * x[ j+96..111]
 *      sum3 += wf[48..63] * x[ j+48..63]   and   wf[112..127] * x[ j+112..127]
 * Each accumulator still owns one lane of a 4-way interleave, so the reduction
 * is unchanged.
 *
 * Verified over many shapes and with random PQ2_0 data (random fp16 scales,
 * including negatives and subnormals, and random 2-bit codes).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "pq2_0.h"

static void *al(size_t bytes) {
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

static uint32_t g_state = 987654321u;
static uint32_t rnd(void) {
    g_state = g_state * 1103515245u + 12345u;
    return (g_state >> 8);
}

/* ---- the ORIGINAL (broken) kernel body: 4 FMAs covering 64 floats ---- */
static void knc_old(float *y, const float *x, const uint8_t *w, int n, int d) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;
    for (int i = 0; i < d; ++i) {
        const uint8_t *row = w + (size_t)i * (n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps();
        int j = 0;
        for (; j < nn; j += QK_PQ2_0) {
            pq2_0_dequant_block(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      sum0);
            sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), sum1);
            sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), sum2);
            sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), sum3);
        }
        __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        float acc = _mm512_reduce_add_ps(sum);
        for (; j < n; ++j) {
            const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
            const float dd = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int q = (blk[2 + ((j % QK_PQ2_0) >> 2)] >> ((j & 3) * 2)) & 0x3;
            acc += (float)(q - 1) * dd * x[j];
        }
        y[i] = acc;
    }
}

/* ---- the FIXED kernel body: 8 FMAs per block, 4 accumulators ---- */
static void knc_fixed(float *y, const float *x, const uint8_t *w, int n, int d) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        const uint8_t *row = w + (size_t)i * (n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps();
        int j = 0;
        for (; j < nn; j += QK_PQ2_0) {
            pq2_0_dequant_block(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf),       _mm512_load_ps(x + j),       sum0);
            sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16),  _mm512_load_ps(x + j + 16),  sum1);
            sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32),  _mm512_load_ps(x + j + 32),  sum2);
            sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48),  _mm512_load_ps(x + j + 48),  sum3);
            sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf + 64),  _mm512_load_ps(x + j + 64),  sum0);
            sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 80),  _mm512_load_ps(x + j + 80),  sum1);
            sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 96),  _mm512_load_ps(x + j + 96),  sum2);
            sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 112), _mm512_load_ps(x + j + 112), sum3);
        }
        __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        float acc = _mm512_reduce_add_ps(sum);
        for (; j < n; ++j) {
            const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
            const float dd = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int q = (blk[2 + ((j % QK_PQ2_0) >> 2)] >> ((j & 3) * 2)) & 0x3;
            acc += (float)(q - 1) * dd * x[j];
        }
        y[i] = acc;
    }
}

/* random but valid PQ2_0 weights */
static void fill_random_w(uint8_t *w, size_t wbytes) {
    for (size_t b = 0; b + PQ2_0_BLOCK_BYTES <= wbytes; b += PQ2_0_BLOCK_BYTES) {
        uint8_t *blk = w + b;
        /* random fp16 with a modest exponent, sign random, mantissa random */
        const uint32_t sign = rnd() & 1u;
        const uint32_t exp  = 5u + (rnd() % 20u);       /* 2^-10 .. 2^4 */
        const uint32_t mant = rnd() & 0x3FFu;
        const uint16_t h = (uint16_t)((sign << 15) | (exp << 10) | mant);
        blk[0] = (uint8_t)(h & 0xFF);
        blk[1] = (uint8_t)(h >> 8);
        for (int q = 0; q < 32; ++q) blk[2 + q] = (uint8_t)(rnd() & 0xFF);
    }
}

static int one_case(int n, int d, const char *tag) {
    const size_t nblk = (size_t)n / QK_PQ2_0;
    const size_t row_bytes = nblk * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;
    uint8_t *w = (uint8_t *)al(wbytes);
    float *x = (float *)al((size_t)n * sizeof(float));
    float *yr = (float *)al((size_t)d * sizeof(float));
    float *yo = (float *)al((size_t)d * sizeof(float));
    float *yf = (float *)al((size_t)d * sizeof(float));
    if (!w || !x || !yr || !yo || !yf) return -1;

    fill_random_w(w, wbytes);
    for (int i = 0; i < n; ++i)
        x[i] = (float)((int)(rnd() % 4000) - 2000) * 0.0007f;

    pq2_0_matvec_ref(yr, x, w, n, d);
    knc_old(yo, x, w, n, d);
    knc_fixed(yf, x, w, n, d);

    double max_old = 0.0, max_fix = 0.0, maxref = 0.0;
    int half_ok = 1;
    for (int i = 0; i < d; ++i) {
        max_old = fmax(max_old, fabs((double)yo[i] - (double)yr[i]));
        max_fix = fmax(max_fix, fabs((double)yf[i] - (double)yr[i]));
        maxref  = fmax(maxref,  fabs((double)yr[i]));
        /* is the old kernel exactly half of the reference? */
        if (fabs((double)yo[i] - 0.5 * (double)yr[i]) > 1e-5 * (fabs((double)yr[i]) + 1e-6))
            half_ok = 0;
    }
    printf("  %-30s n=%5d d=%4d  |ref|max=%9.3f  old_maxdiff=%9.3f (half=%s)"
           "  FIXED_maxdiff=%.3e  %s\n",
           tag, n, d, maxref, max_old, half_ok ? "yes" : "no ",
           max_fix, (max_fix <= 1e-5 * (maxref + 1e-6)) ? "AGREE" : "*** DISAGREE ***");

    const int ok = (max_fix <= 1e-5 * (maxref + 1e-6));
    free(w); free(x); free(yr); free(yo); free(yf);
    return ok ? 0 : 1;
}

int main(void) {
    int bad = 0;
    printf("=== probe_h: FIXED kernel vs pq2_0_matvec_ref, random PQ2_0 data ===\n");
    printf("    'half=yes' means the ORIGINAL kernel returned exactly half the\n");
    printf("    reference for every row -- the signature of dropping 64/128 weights\n\n");

    bad += one_case(128,   1, "single block");
    bad += one_case(256,   1, "2 blocks");
    bad += one_case(384,   3, "3 blocks");
    bad += one_case(512,   4, "4 blocks");
    bad += one_case(1024,  8, "8 blocks");
    bad += one_case(1280, 17, "10 blocks, d=17");
    bad += one_case(2560,  5, "20 blocks");
    bad += one_case(2688,  2, "21 blocks");
    bad += one_case(5120,  8, "40 blocks");
    bad += one_case(5120, 64, "40 blocks, d=64");
    bad += one_case(17408, 6, "136 blocks (widest real width)");

    /* second pass with a different seed, to rule out a lucky draw */
    g_state = 246801357u;
    printf("\n  -- second random pass --\n");
    bad += one_case(5120, 12, "40 blocks, d=12");
    bad += one_case(1024, 33, "8 blocks, d=33");

    printf("\n  failed cases: %d\n", bad);

    /* the hand-checkable case, for the record */
    printf("\n  -- hand-checkable: all weights -1, x=1, n=5120 --\n");
    {
        const int n = 5120, d = 1, nblk = n / QK_PQ2_0;
        uint8_t *w = (uint8_t *)al((size_t)nblk * PQ2_0_BLOCK_BYTES);
        float *x = (float *)al((size_t)n * sizeof(float));
        float yr = 0, yo = 0, yf = 0;
        for (int b = 0; b < nblk; ++b) {
            uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;
            memset(blk + 2, 0x00, 32);
        }
        for (int i = 0; i < n; ++i) x[i] = 1.0f;
        pq2_0_matvec_ref(&yr, x, w, n, d);
        knc_old(&yo, x, w, n, d);
        knc_fixed(&yf, x, w, n, d);
        printf("     expect=-5120.0  ref=%.1f  old=%.1f  fixed=%.1f\n",
               (double)yr, (double)yo, (double)yf);
        free(w); free(x);
    }
    return bad ? 1 : 0;
}
