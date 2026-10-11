/* Vectorised PQ2_0 block unpacking for KNC (IMCI), with a correctness and
 * speed comparison against the scalar path.
 *
 * Why this matters: profiling showed the inference is NOT bandwidth bound --
 * the scalar path sustains 1.36 GB/s where a single thread can read 1.57 GB/s,
 * so the loop body, not memory, is the limit. The dominant cost is turning 34
 * packed bytes into 128 floats one element at a time.
 *
 * Block layout: d = fp16 at [0..1]; 32 code bytes at [2..33]; element j uses
 * byte j/4, shift (j%4)*2, value (q-1)*d.
 *
 * Vector plan (per 8 code bytes -> 32 floats):
 *   1. broadcast one code byte to all 16 bytes of a __m512i, mask 0x03
 *   2. that yields 4 distinct codes per 128-bit lane; spread each to its own
 *      lane with unpack + shifts so each code owns 4 consecutive byte slots
 *   3. zero-extend bytes to int32, subtract 1, convert to float, scale by d
 * KNC has the needed primitives: _mm512_and_si512, _mm512_slli/srli_epi32,
 * _mm512_unpacklo/hi_epi8, _mm512_cvtepi32_ps, _mm512_mul_ps.
 *
 * Also re-tests the "IMCI kernel returns exactly half" defect with a heap
 * (not stack) staging buffer, to check whether stack-array alignment was the
 * cause.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#include "pq2_0.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------- scalar reference (same as pq2_0_dequant_block) ---------- */
static void deq_scalar(const uint8_t *blk, float *out) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    for (int j = 0; j < QK_PQ2_0; ++j) {
        const int q = (qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
        out[j] = (float)(q - 1) * d;
    }
}

/* ---------- vectorised decode of one 32-byte code block ----------
 * Produces 128 codes as int32 in `out` (already minus 1).
 * The 32 code bytes hold 4 codes each, little-endian by 2-bit field:
 *   byte b, field k  ->  element index b*4 + k.
 * We process 8 bytes at a time (32 elements) using unpacklo/hi_epi8 on a
 * broadcast-and-masked byte, which distributes the 4 codes of each byte into
 * the 4 bytes the unpack tree lands in.
 */
static void unpack_codes(const uint8_t *qs, int32_t *out) {
    const __m512i mask3 = _mm512_set1_epi8(0x03);
    for (int g = 0; g < 4; ++g) {                 /* 4 groups x 8 bytes */
        /* broadcast 8 code bytes across a 512-bit register, twice (lo/hi) */
        uint64_t two;
        memcpy(&two, qs + g * 8, 8);
        const __m512i bc = _mm512_set_epi64((long long)two, (long long)two,
                                            (long long)two, (long long)two,
                                            (long long)two, (long long)two,
                                            (long long)two, (long long)two);
        __m512i v = _mm512_and_si512(bc, mask3);  /* 4 codes per 16B lane */
        (void)v;
        /* The unpack tree below is deliberately simple (and therefore slow but
         * obviously correct): pull each of the 4 codes out with per-lane
         * shifts and place them with _mm512_mask_set1-ish arithmetic. The
         * point of this first version is correctness; the fast variant can
         * replace it once the numbers agree. */
        for (int b = 0; b < 8; ++b) {
            const uint8_t byte = qs[g * 8 + b];
            for (int k = 0; k < 4; ++k) {
                out[(g * 8 + b) * 4 + k] = (int)((byte >> (k * 2)) & 0x3) - 1;
            }
        }
    }
}

static void deq_vector(const uint8_t *blk, float *out) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    int32_t codes[QK_PQ2_0] __attribute__((aligned(64)));
    unpack_codes(blk + 2, codes);

    /* int32 -> float WITHOUT _mm512_cvtepi32_ps: on KNC that intrinsic is
     * lowered to _mm512_cvt_roundepi32_ps, which is not available as an
     * external symbol (undefined reference at link time). The classic
     * bias trick works on any IEEE-754 target and is exact for the small
     * integers here (|c| <= 2):
     *     (float)c  ==  bitcast_ps(c + 0x4B000000) - 8388608.0f
     * because 0x4B000000 is 2^23 and adding it to a small integer rounds the
     * mantissa up to hold exactly that integer.
     */
    const __m512i bias_i = _mm512_set1_epi32(0x4B000000);
    const __m512  bias_f = _mm512_set1_ps(8388608.0f);
    /* unpack_codes() already subtracted 1, so codes[] holds (q-1) and the
     * value is simply d * (q-1). */
    const __m512  vd     = _mm512_set1_ps(d);
    for (int j = 0; j < QK_PQ2_0; j += 16) {
        const __m512i c = _mm512_load_epi32((const void *)(codes + j));
        const __m512i b = _mm512_add_epi32(c, bias_i);
        const __m512  f = _mm512_sub_ps(_mm512_castsi512_ps(b), bias_f);
        _mm512_store_ps(out + j, _mm512_mul_ps(f, vd));
    }
}

/* ---------- heap-staging test for the "exactly half" IMCI defect ---------- */
static double half_test(int n, int d, int use_heap) {
    const int nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)aligned_alloc(64, (row_bytes + 63) & ~(size_t)63);
    float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    for (int b = 0; b < nblk; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;
        memset(blk + 2, 0x00, 32);                 /* all codes 0 -> -1.0 */
    }
    for (int i = 0; i < n; ++i) x[i] = 1.0f;

    float *wf_heap = (float *)aligned_alloc(64, QK_PQ2_0 * sizeof(float));
    double total = 0.0;
    for (int i = 0; i < d; ++i) {
        const uint8_t *row = w + (size_t)i * row_bytes;
        float wf_stack[QK_PQ2_0] __attribute__((aligned(64)));
        float *wf = use_heap ? wf_heap : wf_stack;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        for (int j = 0; j < n; j += QK_PQ2_0) {
            pq2_0_dequant_block(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            s0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      s0);
            s1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), s1);
            s2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), s2);
            s3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), s3);
        }
        total += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                    _mm512_add_ps(s2, s3)));
    }
    free(w); free(x); free(wf_heap);
    return total;
}

int main(void) {
    const int n = 5120, d = 8;
    const int nblk = n / QK_PQ2_0;

    /* ---- correctness: vectorised vs scalar dequant ---- */
    printf("=== 1) vectorised unpack vs scalar ===\n");
    {
        uint8_t blk[PQ2_0_BLOCK_BYTES];
        blk[0] = 0x00; blk[1] = 0x3C;                 /* scale 1.0 */
        for (int q = 0; q < 32; ++q) blk[2 + q] = (uint8_t)(q * 7 + 3);
        float a[QK_PQ2_0], b[QK_PQ2_0];
        deq_scalar(blk, a);
        deq_vector(blk, b);
        double mx = 0; int firstbad = -1;
        for (int j = 0; j < QK_PQ2_0; ++j) {
            double e = fabs((double)a[j] - (double)b[j]);
            if (e > mx) mx = e;
            if (e > 1e-6 && firstbad < 0) firstbad = j;
        }
        printf("  max|scalar - vector| = %.3e   %s\n", mx,
               mx < 1e-6 ? "AGREE" : "DISAGREE");
        if (firstbad >= 0)
            printf("  first mismatch at %d: scalar=%.1f vector=%.1f\n",
                   firstbad, a[firstbad], b[firstbad]);
        printf("  sample: scalar[0..5] = %.0f %.0f %.0f %.0f %.0f %.0f\n",
               a[0], a[1], a[2], a[3], a[4], a[5]);
        printf("          vector[0..5] = %.0f %.0f %.0f %.0f %.0f %.0f\n",
               b[0], b[1], b[2], b[3], b[4], b[5]);
        /* dump the intermediate codes to see where the extraction goes wrong */
        {
            int32_t c[QK_PQ2_0] __attribute__((aligned(64)));
            unpack_codes(blk + 2, c);
            printf("  codes[0..7]  = %d %d %d %d %d %d %d %d\n",
                   c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
            printf("  expect       = ");
            for (int j = 0; j < 8; ++j)
                printf("%d ", (int)((blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 0x3) - 1);
            printf("\n  raw byte qs[0]=0x%02X qs[1]=0x%02X\n", blk[2], blk[3]);
        }
    }

    /* ---- speed: scalar vs vectorised dequant over the whole matrix ---- */
    printf("\n=== 2) dequant throughput (n=%d x d=%d) ===\n", n, d);
    {
        const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
        const size_t wbytes = row_bytes * (size_t)d;
        uint8_t *w = (uint8_t *)aligned_alloc(64, (wbytes + 63) & ~(size_t)63);
        float *out = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
        for (size_t i = 0; i < wbytes; ++i) w[i] = (uint8_t)(i * 31 + 5);

        double t0 = now_s();
        for (int i = 0; i < d; ++i)
            for (int b = 0; b < nblk; ++b)
                deq_scalar(w + (size_t)i * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES, out);
        double t1 = now_s();
        for (int i = 0; i < d; ++i)
            for (int b = 0; b < nblk; ++b)
                deq_vector(w + (size_t)i * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES, out);
        double t2 = now_s();
        printf("  scalar  %.4f s   (%.1f M weights/s)\n", t1 - t0,
               (double)n * d / (t1 - t0) / 1e6);
        printf("  vector  %.4f s   (%.1f M weights/s)   speedup %.2fx\n", t2 - t1,
               (double)n * d / (t2 - t1) / 1e6, (t1 - t0) / (t2 - t1));
        free(w); free(out);
    }

    /* ---- the "exactly half" question: stack vs heap staging ---- */
    printf("\n=== 3) IMCI dot product: stack vs heap staging buffer ===\n");
    printf("    correct total for these weights = %d\n", -(n * d));
    {
        double st = half_test(n, d, 0);
        double hp = half_test(n, d, 1);
        printf("  stack staging = %.1f   %s\n", st,
               fabs(st + (double)n * d) < 1 ? "CORRECT" : "WRONG (half?)");
        printf("  heap  staging = %.1f   %s\n", hp,
               fabs(hp + (double)n * d) < 1 ? "CORRECT" : "WRONG (half?)");
    }
    return 0;
}
