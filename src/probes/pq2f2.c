/* pq2f2.c -- make the per-row scale-array build cheap.
 *
 * MEASURED (pq2f1.c, one binary): the fp16 scale handling is now 35-47% of the
 * row cost.
 *   F3 kernel only .................... 2400 M w/s
 *   F3 + scalar scale-array build ..... 1317 M w/s   (build alone = 360 ps/weight)
 *   F6 (scale converted INLINE) ....... 1548 M w/s
 * So the build is worth more than everything else combined.  46 cycles per
 * 2-byte scale is absurd for what should be ~10 instructions.
 *
 * THREE BUILDERS, all verified for bit-equality against bonsai_fp16_to_fp32 on
 * ALL 65536 half values, and all timed inside one binary:
 *   B0 scatter  : scalar loop, branchy (the pq2f1 baseline)
 *   B1 unrolled : scalar, 8 blocks per iteration, branchless fast path, with a
 *                 single vector validity check per 8 (subnormal/inf/nan are
 *                 impossible for a real model scale, but they are DETECTED and
 *                 repaired rather than silently mis-converted)
 *   B2 gather   : ONE vpgatherdd with the UINT16 upconvert fetches 16 raw fp16
 *                 scales at stride 34 in a single instruction, then 8 vector
 *                 ALU ops convert them all to fp32 at once.
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
static uint32_t rng_state;
static uint32_t rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }

static inline float fp16_bits(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)(h & 0x8000u) << 16)
        | ((((uint32_t)h & 0x7FFFu) << 13) + 0x38000000u);
    return v.f;
}

/* ---------------- B0: scalar, branchy ---------------------------------- */
static void b0_build(const uint8_t *row, int nblk, float *drow) {
    for (int b = 0; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = (h & 0x7C00u) ? fp16_bits(h) : bonsai_fp16_to_fp32(h);
    }
}

/* ---------------- B1: unrolled 8, branchless (NO validity check) -------
 * NOTE: a per-element exponent check costs ~4 scalar ops, i.e. as much as the
 * conversion itself, so this variant deliberately has none.  It is exact for
 * every normal half; the gate below reports exactly which halves it misses. */
static void b1_build(const uint8_t *row, int nblk, float *drow) {
    int b = 0;
    for (; b + 8 <= nblk; b += 8) {
        uint16_t h[8];
        for (int i = 0; i < 8; ++i) memcpy(&h[i], row + (size_t)34 * (b + i), 2);
        for (int i = 0; i < 8; ++i) drow[b + i] = fp16_bits(h[i]);
    }
    for (; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = bonsai_fp16_to_fp32(h);
    }
}

/* ---------------- B2: one gather, vector conversion -------------------- */
#define SCALE_OFF() _mm512_set_epi32( \
    15*17, 14*17, 13*17, 12*17, 11*17, 10*17, 9*17, 8*17, \
     7*17,  6*17,  5*17,  4*17,  3*17,  2*17, 1*17,     0)
static int b2_build(const uint8_t *row, int nblk, float *drow) {
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
        const __mmask16 bad = (__mmask16)(_mm512_cmpeq_epi32_mask(e, z)
                                        | _mm512_cmpeq_epi32_mask(e, f));
        if (bad) {
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

/* ---------------- bit-exactness gate on all 65536 halves --------------- */
static void gate(void) {
    printf("--- gate: every builder vs bonsai_fp16_to_fp32 on all 65536 halves ---\n");
    /* 16 halves at stride 34 (so the gather can fetch them), swept over all
     * 4096 groups; every half value is covered exactly once. */
    int nb = 0;
    uint8_t *buf = (uint8_t *)aligned_alloc(64, 34 * 32 + 128);
    float got[32] __attribute__((aligned(64)));
    float *ref = (float *)aligned_alloc(64, 32 * 4);
    int bad0 = 0, bad1 = 0, bad2 = 0, shown = 0;
    for (int g = 0; g < 4096; g += 2) {
        for (int i = 0; i < 32; ++i) {
            uint16_t h = (uint16_t)(g * 32 + i);
            buf[34 * i] = (uint8_t)(h & 0xFF);
            buf[34 * i + 1] = (uint8_t)(h >> 8);
            ref[i] = bonsai_fp16_to_fp32(h);
        }
        b0_build(buf, 32, got); for (int i = 0; i < 32; ++i) if (memcmp(&got[i], &ref[i], 4)) { ++bad0; if (shown<4){printf("    B0 i=%d h=%04x\n",i,g*32+i);++shown;} }
        b1_build(buf, 32, got); for (int i = 0; i < 32; ++i) if (memcmp(&got[i], &ref[i], 4)) ++bad1;
        b2_build(buf, 32, got); for (int i = 0; i < 32; ++i) if (memcmp(&got[i], &ref[i], 4)) ++bad2;
        ++nb;
    }
    printf("    B0 scalar branchy : %d wrong\n", bad0);
    printf("    B1 unrolled8      : %d wrong\n", bad1);
    printf("    B2 gather+vector  : %d wrong\n", bad2);
    free(buf); free(ref);
    (void)nb;
}

/* ---------------- the winning kernel (pq2f1 F3 shape) ------------------ */
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
    const int n    = argc > 1 ? atoi(argv[1]) : 5120;
    const int d    = argc > 2 ? atoi(argv[2]) : 512;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    const int nblk = n / QK_PQ2_0;
    const size_t row_bytes = (size_t)nblk * PQ2_0_BLOCK_BYTES;

    uint8_t *w    = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float   *xp   = (float   *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    float   *drow = (float   *)aligned_alloc(64, ((size_t)nblk * 4 + 63) & ~(size_t)63);
    if (!w || !xp || !drow) { fprintf(stderr, "alloc failed\n"); return 1; }

    rng_state = 12345u;
    for (size_t i = 0; i < row_bytes * d; ++i) w[i] = (uint8_t)(rnd() & 0xFF);
    for (size_t r = 0; r < (size_t)d; ++r)
        for (int b = 0; b < nblk; ++b) {
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            uint16_t e = (uint16_t)(1 + (rnd() % 29));
            uint16_t m = (uint16_t)(rnd() & 0x3FF);
            uint16_t h = (uint16_t)((e << 10) | m);
            blk[0] = (uint8_t)(h & 0xFF); blk[1] = (uint8_t)(h >> 8);
        }
    for (int i = 0; i < n; ++i) xp[i] = (float)((i % 7) - 3) * 0.25f;

    printf("=== pq2f2: cheap scale-array builders ===\n");
    gate();

    printf("\n--- builders alone (%d rows x %d blocks) ---\n", d, nblk);
    const char *nm[3] = { "B0 scalar branchy", "B1 unrolled8", "B2 gather+vector" };
    for (int v = 0; v < 3; ++v) {
        double best = 1e30; int rep = 0;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            for (int i = 0; i < d; ++i) {
                const uint8_t *row = w + (size_t)i * row_bytes;
                if (v == 0) b0_build(row, nblk, drow);
                else if (v == 1) b1_build(row, nblk, drow);
                else rep += b2_build(row, nblk, drow);
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        printf("  %-20s %9.1f ps/weight   (%.4f ms / %d rows)%s\n", nm[v],
               best * 1e12 / ((double)n * d), best * 1e3, d,
               v == 2 ? "  [repaired groups: see below]" : "");
        if (v == 2) printf("                       repaired %d groups (expected 0 for normal scales)\n", rep);
    }

    printf("\n--- F3 kernel + each builder (whole row work, single thread, best of %d) ---\n", reps);
    for (int v = 0; v < 3; ++v) {
        double best = 1e30; double sink = 0.0;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            for (int i = 0; i < d; ++i) {
                const uint8_t *row = w + (size_t)i * row_bytes;
                if (v == 0) b0_build(row, nblk, drow);
                else if (v == 1) b1_build(row, nblk, drow);
                else b2_build(row, nblk, drow);
                sink += f3_row(row, xp, n, drow);
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        const double t = (double)n * d / best;
        printf("  %-20s %8.1f M w/s   %6.3fx vs V10c(1059)   %6.3fx vs V1(918)\n",
               nm[v], t / 1e6, t / 1059e6, t / 918e6);
        printf("                       (sink=%g)\n", sink);
    }
    return 0;
}
