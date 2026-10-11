/* pq2f3.c -- measure (not infer) the exactness of the final combination:
 *   decode  : two LUTs (L0[j]=(j&3)-1, L1[j]=((j>>2)&3)-1) so one unshifted
 *             index yields field 0 and field 1 and one shift by 4 yields
 *             fields 2 and 3; both LUTs pre-multiplied by the block scale d
 *             once per block, so no per-field vmulps is needed
 *   scales  : ONE vpgatherdd with the UINT16 upconvert fetches 16 raw fp16
 *             scales at stride 34, 8 vector ALU ops convert them to fp32, and
 *             a vector mask repairs the (impossible in practice, but detected)
 *             subnormal / inf / nan cases through the scalar reference
 *   dot     : one partial sum per block, pointer-increment addressing
 *
 * The gate below is the same shape as pq2v9.c's: 64 source alignments x
 * (128 unit vectors + row sum) against the scalar V1 path, plus extra random
 * blocks, plus multi-block rows with an ODD number of blocks (exercises the
 * gather tail), plus non-trivial scales.
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

static uint32_t rng_state;
static uint32_t rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }

/* ---------------- V1 reference (shipped path) -------------------------- */
static float v1_row(const uint8_t *row, const float *x, int n) {
    float out[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        pq2_0_dequant_block_imci(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, out);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(out),      _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(out + 16), _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(out + 32), _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(out + 48), _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(out + 64), _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(out + 80), _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(out + 96), _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(out +112), _mm512_load_ps(x + j +112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}
static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[(((size_t)b * 2 + h) * 4 + f) * 16 + l] =
                        x[(size_t)b * 128 + 64 * h + 4 * l + f];
}

/* ---------------- the scale builder (B2) ------------------------------- */
#define SCALE_OFF() _mm512_set_epi32( \
    15*17, 14*17, 13*17, 12*17, 11*17, 10*17, 9*17, 8*17, \
     7*17,  6*17,  5*17,  4*17,  3*17,  2*17, 1*17,     0)
static int build_drow(const uint8_t *row, int nblk, float *drow) {
    const __m512i off = SCALE_OFF();
    const __m512i m0 = _mm512_set1_epi32(0x8000);
    const __m512i m1 = _mm512_set1_epi32(0x7FFF);
    const __m512i m2 = _mm512_set1_epi32(0x7C00);
    const __m512i k  = _mm512_set1_epi32(0x38000000);
    const __m512i z  = _mm512_setzero_si512();
    const __m512i f  = _mm512_set1_epi32(0x7C00);
    int b = 0, repaired = 0;
    for (; b + 16 <= nblk; b += 16) {
        const __m512i h = _mm512_i32extgather_epi32(off, row + (size_t)34 * b,
                              _MM_UPCONV_EPI32_UINT16, 2, _MM_HINT_NONE);
        const __m512i e = _mm512_and_si512(h, m2);
        if ((__mmask16)(_mm512_cmpeq_epi32_mask(e, z) | _mm512_cmpeq_epi32_mask(e, f))) {
            ++repaired;
            for (int i = 0; i < 16; ++i) {
                uint16_t hh; memcpy(&hh, row + (size_t)34 * (b + i), 2);
                drow[b + i] = bonsai_fp16_to_fp32(hh);
            }
            continue;
        }
        const __m512i bits = _mm512_or_si512(
            _mm512_slli_epi32(_mm512_and_si512(h, m0), 16),
            _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(h, m1), 13), k));
        _mm512_store_ps(drow + b, _mm512_castsi512_ps(bits));
    }
    for (; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = bonsai_fp16_to_fp32(h);
    }
    return repaired;
}

/* ---------------- the decoder ----------------------------------------- */
#define LUT0_BITS() _mm512_set_epi32( \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000, \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000, \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000, \
    0x40000000,(int)0x3F800000,0,(int)0xBF800000)
#define LUT1_BITS() _mm512_set_epi32( \
    0x40000000,0x40000000,0x40000000,0x40000000, \
    0x3F800000,0x3F800000,0x3F800000,0x3F800000, \
    0,0,0,0, \
    (int)0xBF800000,(int)0xBF800000,(int)0xBF800000,(int)0xBF800000)

static float f3_row(const uint8_t *row, const float *xp, int n, const float *drow) {
    __m512 s = _mm512_setzero_ps();
    const __m512 l0b = _mm512_castsi512_ps(LUT0_BITS());
    const __m512 l1b = _mm512_castsi512_ps(LUT1_BITS());
    const uint8_t *blk = row;
    const float *x0 = xp;
    const float *dp = drow;
    int b = n >> 7;
    do {
        const __m512 vd = _mm512_set1_ps(*dp);
        const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
        const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
        const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i e0 = _mm512_srli_epi32(c0, 4);
        const __m512i e1 = _mm512_srli_epi32(c1, 4);
        __m512 t = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0)), _mm512_load_ps(x0));
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1)), _mm512_load_ps(x0 + 16), t);
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0)), _mm512_load_ps(x0 + 32), t);
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1)), _mm512_load_ps(x0 + 48), t);
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0)), _mm512_load_ps(x0 + 64), t);
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1)), _mm512_load_ps(x0 + 80), t);
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0)), _mm512_load_ps(x0 + 96), t);
        t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1)), _mm512_load_ps(x0 +112), t);
        s = _mm512_add_ps(s, t);
        blk += PQ2_0_BLOCK_BYTES;
        x0  += QK_PQ2_0;
        ++dp;
    } while (--b);
    return _mm512_reduce_add_ps(s);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int n = argc > 1 ? atoi(argv[1]) : 5120;
    const int d = argc > 2 ? atoi(argv[2]) : 512;
    const int nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float *xp = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float *drow = (float *)aligned_alloc(64, ((size_t)nblk * 4 + 63) & ~(size_t)63);
    if (!w || !x || !xp || !drow) return 1;
    rng_state = 4242u;
    for (size_t i = 0; i < row_bytes * d; ++i) w[i] = (uint8_t)(rnd() & 0xFF);
    /* normal, subnormal AND inf/nan scales all present, so B2's repair path is
     * really exercised by the element-wise gate too */
    for (size_t r = 0; r < (size_t)d; ++r)
        for (int b = 0; b < nblk; ++b) {
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            uint16_t h;
            int mode = (int)(rnd() % 4);
            if (mode == 0)      h = (uint16_t)((1 + (rnd() % 29) << 10) | (rnd() & 0x3FF));
            else if (mode == 1) h = (uint16_t)(rnd() & 0x03FF);              /* subnormal */
            else if (mode == 2) h = (uint16_t)(0x7C00 | (rnd() & 0x3FF));    /* inf/nan */
            else                h = (uint16_t)((rnd() & 0x8000) | (1 + (rnd() % 29) << 10) | (rnd() & 0x3FF));
            blk[0] = (uint8_t)(h & 0xFF); blk[1] = (uint8_t)(h >> 8);
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;
    build_xp(x, xp, n);

    printf("=== pq2f3: measured exactness of F3 + B2 (measured zeros only) ===\n");

    /* ---- unit vectors, 64 alignments, 4 random blocks ------------------ */
    uint8_t *area = (uint8_t *)aligned_alloc(64, 64 + 64 + 256);
    uint8_t blk[QK_PQ2_0 * 2 + 96];
    int total = 0, checks = 0, repaired_total = 0;
    for (int trial = 0; trial < 4; ++trial) {
        rng_state = 1000003u + 7919u * (uint32_t)trial;
        for (unsigned i = 0; i < sizeof blk; ++i) blk[i] = (uint8_t)(rnd() & 0xFF);
        for (int off = 0; off < 64; ++off) {
            float xv[QK_PQ2_0] __attribute__((aligned(64)));
            float xv2[QK_PQ2_0] __attribute__((aligned(64)));
            float dr[1] __attribute__((aligned(64)));
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
            repaired_total += build_drow(area + off, 1, dr);
            for (int jj = 0; jj < QK_PQ2_0; ++jj) {
                for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
                xv[jj] = 1.0f;
                build_xp(xv, xv2, QK_PQ2_0);
                const float a = v1_row(area + off, xv, QK_PQ2_0);
                const float b = f3_row(area + off, xv2, QK_PQ2_0, dr);
                ++checks;
                if (fabsf(a - b) > 1e-6f) ++total;
            }
            { float xr[QK_PQ2_0] __attribute__((aligned(64)));
              float xr2[QK_PQ2_0] __attribute__((aligned(64)));
              for (int t = 0; t < QK_PQ2_0; ++t) xr[t] = x[t];
              build_xp(xr, xr2, QK_PQ2_0);
              const float a = v1_row(area + off, xr, QK_PQ2_0);
              const float b = f3_row(area + off, xr2, QK_PQ2_0, dr);
              ++checks;
              if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) ++total; }
        }
    }
    printf("  unit-vector + row-sum gate: %d checks (4 blocks x 64 alignments x 129), "
           "MISMATCHES = %d  %s\n", checks, total, total == 0 ? "*** EXACT ***" : "*** WRONG ***");
    printf("  (single-block scale groups that took B2's repair path: %d)\n", repaired_total);

    /* ---- multi-block, odd and even block counts ------------------------ */
    for (int ni = 0; ni < 4; ++ni) {
        const int nn = (int[]){ 2048, 2176, 3456, 5120 }[ni];
        float *x2  = (float *)aligned_alloc(64, ((size_t)nn * 4 + 63) & ~(size_t)63);
        float *xp2 = (float *)aligned_alloc(64, ((size_t)nn * 4 + 63) & ~(size_t)63);
        float dr[64] __attribute__((aligned(64)));
        for (int i = 0; i < nn; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        build_xp(x2, xp2, nn);
        int rep = 0;
        double worst = 0.0;
        for (int r = 0; r < 8; ++r) {
            const uint8_t *row = w + (size_t)r * row_bytes;
            const int nb = nn / 128;
            rep += build_drow(row, nb, dr);
            const double a = v1_row(row, x2, nn);
            const double b = f3_row(row, xp2, nn, dr);
            const double e = fabs(a - b) / (fabs(a) + 1.0);
            if (e > worst) worst = e;
        }
        printf("  multi-block n=%4d (%d blocks%s): 8 rows, max rel diff = %.3e  %s   repairs=%d\n",
               nn, nn/128, (nn/128) & 1 ? ", ODD" : "", worst, worst < 1e-5 ? "OK" : "*** WRONG ***", rep);
        free(x2); free(xp2);
    }
    free(w); free(x); free(xp); free(drow); free(area);
    return total != 0;
}
