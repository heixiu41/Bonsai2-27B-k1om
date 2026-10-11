/* pq2imci.c -- correctness + throughput of the vectorised PQ2_0 decoder
 * (pq2_0_dequant_block_imci) against the scalar pq2_0_dequant_block, run on
 * the KNC card.
 *
 * Sections:
 *   1) instruction semantics: does the unaligned extload path behave?
 *   2) element-by-element block comparison over random data and every source
 *      alignment (0..63), reporting max |diff| and the first mismatch
 *   3) whole-matrix decode comparison (10240 x 5120 = 52,428,800 weights)
 *   4) single-thread decode throughput, scalar vs vectorised
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

static uint32_t rng = 12345u;
static uint32_t xr(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

/* a random but finite fp16: keeps exp < 0x1F so no inf/nan, exercises
 * subnormals (exp == 0) and negatives. */
static uint16_t rand_fp16(void) {
    uint16_t sign = (uint16_t)(xr() & 1) << 15;
    uint32_t e;
    uint32_t r = xr() % 100;
    if (r < 10)      e = 0;                 /* subnormal / zero */
    else if (r < 25) e = 1 + xr() % 3;      /* tiny */
    else             e = 1 + xr() % 30;     /* normal-ish */
    return (uint16_t)(sign | (e << 10) | (xr() & 0x3FF));
}

static void fill_block(uint8_t *blk) {
    const uint16_t h = rand_fp16();
    blk[0] = (uint8_t)(h & 0xFF);
    blk[1] = (uint8_t)(h >> 8);
    for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(xr() & 0xFF);
}

/* A deliberately un-optimisable scalar reference: volatile forces every
 * operation to be emitted, so this cannot be auto-vectorised or reassociated.
 * Used to decide WHICH of the two implementations is wrong when they differ. */
static void deq_naive(const uint8_t *blk, float *out) {
    volatile float d;
    volatile int   q;
    volatile float r;
    d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    for (int j = 0; j < QK_PQ2_0; ++j) {
        q = (int)(((unsigned)blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 0x3);
        r = (float)(q - 1) * d;
        out[j] = r;
    }
}

/* Variant B: same extraction, but the int->float step uses the IEEE-754 bias
 * identity instead of the 16-entry LUT:
 *     (float)k == bitcast_ps(k + 0x4B000000) - 8388608.0f     for 0 <= k < 2^23
 * Here k = code & 3 is always 0..3, so there is no negative operand and the
 * identity is exact; folding the -1 into the constant subtract gives
 *     (q - 1) == bitcast_ps((code & 3) + 0x4B000000) - 8388609.0f
 * This trades one (slow) vpermd for three cheap integer/FP ops per 16 weights. */
static inline void deq_imci_bias(const uint8_t *blk, float *out) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    const __m512i shf    = PQ2_IMCI_SHIFT();
    const __m512i three  = _mm512_set1_epi32(3);
    const __m512i bias_i = _mm512_set1_epi32(0x4B000000);
    const __m512  bias_f = _mm512_set1_ps(8388609.0f);   /* 2^23 + 1 */
    const __m512  vd     = _mm512_set1_ps(d);
    const __m512i c0 = PQ2_IMCI_LOAD16(qs);
    const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
    int m;
    for (m = 0; m < 8; ++m) {
        const __m512i c = (m < 4) ? c0 : c1;
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m & 3), c);
        const __m512i v = _mm512_srlv_epi32(b, shf);
        const __m512i k = _mm512_and_si512(v, three);            /* code 0..3 */
        const __m512i t = _mm512_add_epi32(k, bias_i);           /* + 2^23 */
        const __m512  g = _mm512_sub_ps(_mm512_castsi512_ps(t), bias_f);
        _mm512_store_ps(out + 16 * m, _mm512_mul_ps(g, vd));
    }
}

/* ---------- 1) unaligned widening-load semantics ----------
 * The DECODER's load primitive is PQ2_IMCI_LOAD16 = lo(p)+hi(p+64) in UINT8
 * mode: 16 bytes at the arbitrary address p -> 16 dwords, dword i = p[i].
 * Neither half works alone: lo() zero-fills everything past the 64-byte aligned
 * block containing p (verified in pq2load3.c).  The aligned forms fault when
 * misaligned, and the NONE (packed dword) form of the same intrinsic needs
 * 4-byte alignment, which qs = blk+2 does not have. */
static int test_extload(void) {
    uint8_t buf[256] __attribute__((aligned(64)));
    int bad = 0;
    printf("=== 1) _mm512_extloadunpacklo/hi_epi32 UINT8: 16 bytes at any offset ===\n");
    for (int i = 0; i < 256; ++i) buf[i] = (uint8_t)(i * 7 + 1);
    for (int off = 0; off < 64; ++off) {          /* every byte offset, incl. odd */
        int32_t got[16] __attribute__((aligned(64)));
        const __m512i v = PQ2_IMCI_LOAD16(buf + off);
        _mm512_store_epi32((void *)got, v);
        for (int i = 0; i < 16; ++i) {
            if (got[i] != (int32_t)buf[off + i]) {
                printf("  offset %2d lane %2d: MISMATCH got %d want %d\n",
                       off, i, got[i], (int)buf[off + i]);
                ++bad;
            }
        }
    }
    printf("  offsets 0..63 x 16 lanes  : %s\n", bad ? "FAILED" :
           "all correct (including every 64-byte boundary crossing)");
    /* record what the lo form alone does: this was the original bug */
    {
        int32_t got[16] __attribute__((aligned(64)));
        __m512i v = _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(),
                        (const void *)(buf + 52), _MM_UPCONV_EPI32_UINT8,
                        _MM_HINT_NONE);
        _mm512_store_epi32((void *)got, v);
        printf("  lo(+52) ALONE             : %d %d %d %d %d %d | %d %d %d %d %d %d "
               "<- lanes 12..15 are ZERO\n",
               got[0], got[1], got[2], got[3], got[4], got[5],
               got[10], got[11], got[12], got[13], got[14], got[15]);
        printf("                              (buf[52..63] = 109 116 .. 144; "
               "buf[64..67] = 193 200 207 214 were silently lost)\n");
    }
    return bad;
}

/* ---------- 2) element-by-element block comparison ---------- */
static int test_blocks(int nblocks) {
    uint8_t  *arena;
    float    *a, *b, *c;
    int worst_align_bad = 0;
    double maxdiff = 0.0, maxdiff2 = 0.0;
    long   total = 0, mism = 0, mism2 = 0;
    int    firstbad = -1, firstbad_idx = -1;
    double first_a = 0, first_b = 0;

    arena = (uint8_t *)aligned_alloc(64, 4096);
    a = (float *)aligned_alloc(64, QK_PQ2_0 * sizeof(float));
    b = (float *)aligned_alloc(64, QK_PQ2_0 * sizeof(float));
    c = (float *)aligned_alloc(64, QK_PQ2_0 * sizeof(float));
    if (!arena || !a || !b || !c) { printf("alloc failed\n"); exit(1); }
    memset(arena, 0, 4096);

    printf("\n=== 2) pq2_0_dequant_block_imci vs pq2_0_dequant_block ===\n");
    printf("  %d random blocks x 64 source alignments (block start 0..63,\n"
           "  so qs covers every residue mod 64 including the boundary cases)\n", nblocks);
    rng = 99991u;

    for (int blk_i = 0; blk_i < nblocks; ++blk_i) {
        uint8_t tmp[PQ2_0_BLOCK_BYTES];
        fill_block(tmp);
        for (int al = 0; al < 64; ++al) {
            uint8_t *p = arena + 128 + al;         /* deliberately misaligned */
            memcpy(p, tmp, PQ2_0_BLOCK_BYTES);
            pq2_0_dequant_block(p, a);
            pq2_0_dequant_block_imci(p, b);
            deq_imci_bias(p, c);
            for (int j = 0; j < QK_PQ2_0; ++j) {
                const double e  = fabs((double)a[j] - (double)b[j]);
                const double e2 = fabs((double)a[j] - (double)c[j]);
                if (e  > maxdiff)  maxdiff  = e;
                if (e2 > maxdiff2) maxdiff2 = e2;
                if (e  != 0.0) {
                    ++mism;
                    if (firstbad < 0) {
                        firstbad = blk_i; firstbad_idx = j;
                        first_a = a[j]; first_b = b[j];
                    }
                }
                if (e2 != 0.0) ++mism2;
                ++total;
            }
        }
        (void)worst_align_bad;
    }

    printf("  elements compared      : %ld\n", total);
    printf("  max |scalar - vector(LUT)|  : %.17g   mismatches %ld\n", maxdiff,  mism);
    printf("  max |scalar - vector(BIAS)| : %.17g   mismatches %ld\n", maxdiff2, mism2);
    if (firstbad >= 0)
        printf("  first mismatch         : block %d elem %d  scalar=%.9g vector=%.9g\n",
               firstbad, firstbad_idx, first_a, first_b);
    printf("  verdict                : %s\n",
           (mism == 0 && mism2 == 0) ? "BOTH EXACT (bit-identical)"
                                     : (mism == 0 ? "LUT exact, BIAS wrong"
                                                  : "LUT wrong"));

    free(arena); free(a); free(b); free(c);
    return (int)(mism + mism2);
}

/* ---------- 3+4) whole matrix ---------- */
static void test_matrix(int n, int d) {
    const int    nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    const size_t wbytes = row_bytes * (size_t)d;
    uint8_t *w;
    float *o1, *o2;
    double t0, t1, ts, tv, tb;
    int reps;
    long total_mism = 0;
    double maxdiff = 0.0;
    int firstbad_row = -1, firstbad_elem = -1;
    double firstbad_a = 0, firstbad_b = 0;

    /* +512 slack: the lo/hi pair reads the 64-byte aligned blocks bracketing
     * each payload, so it touches up to ~96 bytes past the last block. */
    w  = (uint8_t *)aligned_alloc(64, ((wbytes + 512) + 63) & ~(size_t)63);
    o1 = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    o2 = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    if (!w || !o1 || !o2) { printf("alloc failed (%zu bytes)\n", wbytes); exit(1); }

    /* Fill with FINITE fp16 scales (raw random bytes would give inf/nan
     * exponents, and nan != nan would be counted as a mismatch). */
    rng = 4242u;
    for (size_t b = 0; b < (size_t)d * nblk; ++b) fill_block(w + b * PQ2_0_BLOCK_BYTES);

    /* --- exactness over the whole matrix: scalar vs naive vs vector --- */
    printf("\n=== 3) whole-matrix decode  n=%d d=%d  (%d weights, %.2f MB packed) ===\n",
           n, d, n * d, (double)wbytes / 1048576.0);
    {
        long   m_sv = 0, m_nv = 0, m_sn = 0;
        double mx_sv = 0.0;
        int    shown = 0;
        for (int i = 0; i < d; ++i) {
            const uint8_t *row = w + (size_t)i * row_bytes;
            for (int b = 0; b < nblk; ++b) {
                const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
                pq2_0_dequant_block(blk, o1 + (size_t)b * QK_PQ2_0);
                pq2_0_dequant_block_imci(blk, o2 + (size_t)b * QK_PQ2_0);
                if (!shown) {
                    /* first block of the run: full three-way cross-check */
                    static float nv[QK_PQ2_0] __attribute__((aligned(64)));
                    deq_naive(blk, nv);
                    for (int j = 0; j < QK_PQ2_0; ++j) {
                        const double e = fabs((double)o1[j] - (double)o2[j]);
                        if (e > mx_sv) mx_sv = e;
                        if (e != 0.0) {
                            ++m_sv;
                            if (m_sv <= 4)
                                printf("  MISMATCH row %d blk %d elem %d: "
                                       "scalar=%.9g naive=%.9g vector=%.9g  "
                                       "qs[%d]=0x%02x d_raw=0x%02x%02x\n",
                                       i, b, j, o1[j], nv[j], o2[j],
                                       (j >> 2), blk[2 + (j >> 2)], blk[1], blk[0]);
                        }
                        if (fabs((double)nv[j] - (double)o2[j]) != 0.0) ++m_nv;
                        if (fabs((double)nv[j] - (double)o1[j]) != 0.0) ++m_sn;
                    }
                    shown = 1;
                } else {
                    for (int j = 0; j < QK_PQ2_0; ++j) {
                        const double e = fabs((double)o1[j] - (double)o2[j]);
                        if (e > mx_sv) mx_sv = e;
                        if (e != 0.0) ++m_sv;
                    }
                }
                if (i == 0 && b == 0) {
                    printf("  block 0 raw d = 0x%02x%02x  qs[0..7] = "
                           "%02x %02x %02x %02x %02x %02x %02x %02x\n",
                           blk[1], blk[0], blk[2],blk[3],blk[4],blk[5],
                           blk[6],blk[7],blk[8],blk[9]);
                }
            }
            if (i > 2) break;             /* 3 rows is plenty for the cross-check */
        }
        /* now the full-matrix scalar-vs-vector count over every row */
        m_sv = 0; mx_sv = 0.0;
        {
            long shown = 0;
            for (int i = 0; i < d; ++i) {
                const uint8_t *row = w + (size_t)i * row_bytes;
                for (int b = 0; b < nblk; ++b) {
                    const uint8_t *blk = row + (size_t)b * PQ2_0_BLOCK_BYTES;
                    pq2_0_dequant_block(blk, o1 + (size_t)b * QK_PQ2_0);
                    pq2_0_dequant_block_imci(blk, o2 + (size_t)b * QK_PQ2_0);
                }
                for (int j = 0; j < n; ++j) {
                    const double e = fabs((double)o1[j] - (double)o2[j]);
                    if (e > mx_sv) mx_sv = e;
                    if (e != 0.0) {
                        ++m_sv;
                        if (shown < 6) {
                            /* re-run just this block with fresh local buffers */
                            static float ls[QK_PQ2_0] __attribute__((aligned(64)));
                            static float ln[QK_PQ2_0] __attribute__((aligned(64)));
                            static float lv[QK_PQ2_0] __attribute__((aligned(64)));
                            const int    bbad = j / QK_PQ2_0;
                            const int    ebad = j % QK_PQ2_0;
                            const uint8_t *pb = w + (size_t)i * row_bytes
                                                + (size_t)bbad * PQ2_0_BLOCK_BYTES;
                            pq2_0_dequant_block(pb, ls);
                            deq_naive(pb, ln);
                            pq2_0_dequant_block_imci(pb, lv);
                            printf("  MISMATCH row %d blk %d elem %d (codebyte %d field %d)\n",
                                   i, bbad, ebad, ebad >> 2, ebad & 3);
                            printf("    main-loop: scalar=%.9g vector=%.9g   (diff %.9g)\n",
                                   o1[j], o2[j], (double)o1[j] - (double)o2[j]);
                            printf("    fresh    : scalar=%.9g naive=%.9g vector=%.9g\n",
                                   ls[ebad], ln[ebad], lv[ebad]);
                            printf("    blockaddr%%64=%d qsaddr%%64=%d  qs[%d]=0x%02x "
                                   "d_raw=0x%02x%02x\n",
                                   (int)((uintptr_t)pb & 63),
                                   (int)((uintptr_t)(pb + 2) & 63),
                                   ebad >> 2, pb[2 + (ebad >> 2)], pb[1], pb[0]);
                            printf("    qs[16..31] =");
                            for (int qq = 16; qq < 32; ++qq) printf(" %02x", pb[2 + qq]);
                            printf("\n");
                            ++shown;
                        }
                    }
                }
            }
        }
        printf("  weights compared       : %d\n", n * d);
        printf("  max |scalar - vector|  : %.17g\n", mx_sv);
        printf("  mismatching weights    : %ld\n", m_sv);
        printf("  (3-row 3-way cross-check: naive-vs-vector %ld, naive-vs-scalar %ld)\n",
               m_nv, m_sn);
        printf("  verdict                : %s\n",
               m_sv == 0 ? "EXACT over the entire matrix" : "NOT EXACT");
    }

    /* --- throughput, single thread --- */
    reps = (n * d < 20000000) ? 60 : 8;
    printf("\n=== 4) single-thread decode throughput (n=%d d=%d, %d reps) ===\n", n, d, reps);

    t0 = now_s();
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < d; ++i) {
            const uint8_t *row = w + (size_t)i * row_bytes;
            for (int b = 0; b < nblk; ++b)
                pq2_0_dequant_block(row + (size_t)b * PQ2_0_BLOCK_BYTES,
                                    o1 + (size_t)b * QK_PQ2_0);
        }
    ts = now_s() - t0;

    t1 = now_s();
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < d; ++i) {
            const uint8_t *row = w + (size_t)i * row_bytes;
            for (int b = 0; b < nblk; ++b)
                pq2_0_dequant_block_imci(row + (size_t)b * PQ2_0_BLOCK_BYTES,
                                         o2 + (size_t)b * QK_PQ2_0);
        }
    tv = now_s() - t1;

    t1 = now_s();
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < d; ++i) {
            const uint8_t *row = w + (size_t)i * row_bytes;
            for (int b = 0; b < nblk; ++b)
                deq_imci_bias(row + (size_t)b * PQ2_0_BLOCK_BYTES,
                              o2 + (size_t)b * QK_PQ2_0);
        }
    tb = now_s() - t1;

    {
        const double nw = (double)n * d * reps;
        const double gb = (double)wbytes * reps / 1e9;
        printf("  scalar       %8.4f s   %8.1f M weights/s   %6.3f GB/s packed\n",
               ts, nw / ts / 1e6, gb / ts);
        printf("  vector LUT   %8.4f s   %8.1f M weights/s   %6.3f GB/s packed"
               "   speedup %.2fx\n",
               tv, nw / tv / 1e6, gb / tv, ts / tv);
        printf("  vector BIAS  %8.4f s   %8.1f M weights/s   %6.3f GB/s packed"
               "   speedup %.2fx\n",
               tb, nw / tb / 1e6, gb / tb, ts / tb);
        printf("  BEST ONE-THREAD SPEEDUP: %.2fx (%s)\n",
               (ts / tv > ts / tb) ? ts / tv : ts / tb,
               (ts / tv > ts / tb) ? "LUT" : "BIAS");
    }

    free(w); free(o1); free(o2);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("PQ2_0 IMCI dequantiser -- KNC (k1om) on-card test\n");
    printf("QK_PQ2_0=%d  block=%d bytes\n\n", QK_PQ2_0, PQ2_0_BLOCK_BYTES);

    /* sample: the exact scenario quoted in the task */
    {
        uint8_t blk[PQ2_0_BLOCK_BYTES];
        float a[QK_PQ2_0] __attribute__((aligned(64)));
        float b[QK_PQ2_0] __attribute__((aligned(64)));
        blk[0] = 0x00; blk[1] = 0x3C;                 /* scale 1.0 */
        for (int q = 0; q < 32; ++q) blk[2 + q] = (uint8_t)(q * 7 + 3);
        pq2_0_dequant_block(blk, a);
        pq2_0_dequant_block_imci(blk, b);
        printf("=== 0) sample block (d = 1.0) ===\n");
        printf("  raw qs[0..3] = 0x%02X 0x%02X 0x%02X 0x%02X\n",
               blk[2], blk[3], blk[4], blk[5]);
        printf("  scalar[0..7] = %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        printf("  vector[0..7] = %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
        printf("  expect       = 2 -1 -1 -1 1 1 -1 -1\n\n");
    }

    test_extload();
    test_blocks(4000);
    /* cache-resident (256 KB packed, fits the 512 KB L2) -> compute-bound */
    test_matrix(1024, 32);
    /* the required full-size matrix -> includes main-memory streaming */
    test_matrix(10240, 5120);
    return 0;
}
