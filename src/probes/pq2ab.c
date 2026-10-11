/* pq2ab.c -- SAME-BINARY A/B: shipped V12 decoder vs the F3+B2 candidate.
 *
 * Cross-binary comparisons have shown 20% phantom differences on this project,
 * so BOTH decoders live in this one binary and the timing runs them in BOTH
 * orders (V12,F3 then F3,V12).  Every timed rep flips a weight byte (defeats
 * CSE/hoisting) and every result is accumulated into a printed volatile sink
 * (defeats dead-code elimination).
 *
 *   gate 1  row level: 4 random blocks x 64 source alignments x (128 unit
 *           vectors + row sum) vs the scalar reference, BOTH decoders
 *   gate 2  forced special scales: min/max subnormal, min/max normal, negative
 *           subnormal, one in the gather TAIL (odd block count); proves B2's
 *           repair path with FINITE values, plus a bit-exact drow check
 *   gate 3  multi-block rows, odd and even block counts, random scales incl.
 *           subnormal / inf / nan
 *   gate 4  matvec level (omp at the current thread count): shipped
 *           pq2_0_matvec_knc and f3_matvec vs pq2_0_matvec_ref
 *   timing  real shape mix (5120x17408 x2, 17408x5120, 5120x10240, 5120x6144),
 *           best-of-N, both orders
 *
 * KNC traps respected: no loadu/storeu, _mm512_load_ps only on 64-byte-aligned
 * buffers, gathers via the proven _mm512_i32extgather_epi32 UINT16 upconvert.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#include "pq2_0.h"

static uint32_t rng_state;
static uint32_t rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static volatile float sink;

/* ---------------- scalar reference row (shipped scalar dequant) ---------- */
static float ref_row(const uint8_t *row, const float *x, int n) {
    float wf[QK_PQ2_0];
    float acc = 0.0f;
    for (int j = 0; j < n; j += QK_PQ2_0) {
        pq2_0_dequant_block(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
        for (int k = 0; k < QK_PQ2_0; ++k) acc += wf[k] * x[j + k];
    }
    return acc;
}

/* ---------------- V12 row: VERBATIM copy of the shipped kernel body ------ */
static float v12_row(const uint8_t *row, const float *xp, int n) {
    const __m512 la0 = pq2_lut_a(), lb0 = pq2_lut_b();
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const size_t b = (size_t)(j / QK_PQ2_0);
        const uint8_t *blk = row + b * PQ2_0_BLOCK_BYTES;
        const float dscale =
            bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const __m512 vd = _mm512_set1_ps(dscale);
        const __m512i c0  = PQ2_IMCI_LOAD16(blk + 2);
        const __m512i c1  = PQ2_IMCI_LOAD16(blk + 18);
        const __m512i c0h = _mm512_srli_epi32(c0, 4);
        const __m512i c1h = _mm512_srli_epi32(c1, 4);
        const float *xp0 = xp + b * QK_PQ2_0;
        const float *xp1 = xp0 + 64;
        const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
        const __m512i lai = _mm512_castps_si512(la);
        const __m512i lbi = _mm512_castps_si512(lb);
        __m512 g[8];
        g[0] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lai));
        g[1] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0,  lbi));
        g[2] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lai));
        g[3] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c0h, lbi));
        g[4] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lai));
        g[5] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1,  lbi));
        g[6] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lai));
        g[7] = _mm512_castsi512_ps(_mm512_permutevar_epi32(c1h, lbi));
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(xp0 +  0), s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(xp0 + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(xp0 + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(xp0 + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(xp1 +  0), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(xp1 + 16), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(xp1 + 32), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(xp1 + 48), s3);
    }
    __m512 sum = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(sum);
}

/* ---------------- F3 candidate: B2 scale builder + decoder (from pq2f3) -- */
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

/* F3 as a full matvec, same shape as pq2_0_matvec_knc: xp built once per
 * call, drow built per row inside the omp loop (thread-safe: stack buffer
 * private to each iteration). */
#define PQ2_MAX_NBLK (PQ2_XP_MAX / QK_PQ2_0)
static void f3_matvec(float *y, const float *x, const uint8_t *w, int n, int d) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;
    const size_t rb = ((size_t)n + QK_PQ2_0 - 1) / QK_PQ2_0 * PQ2_0_BLOCK_BYTES;
    float xp[PQ2_XP_MAX] __attribute__((aligned(64)));
    if (nn != n || n > PQ2_XP_MAX) {
        pq2_0_matvec_ref(y, x, w, n, d);
        return;
    }
    pq2_build_xp(x, xp, n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        float drow[PQ2_MAX_NBLK] __attribute__((aligned(64)));
        const uint8_t *row = w + (size_t)i * rb;
        build_drow(row, nn / QK_PQ2_0, drow);
        y[i] = f3_row(row, xp, nn, drow);
    }
}

/* ---------------- NaN/inf-aware comparisons ------------------------------ */
static int bad_abs(float a, float b, float tol) {
    if (isnan(a) && isnan(b)) return 0;
    if (isinf(a) && isinf(b) && (a > 0) == (b > 0)) return 0;
    return !(fabsf(a - b) <= tol);
}
static int bad_rel(double a, double b, double tol) {
    if (isnan(a) && isnan(b)) return 0;
    if (isinf(a) && isinf(b) && (a > 0) == (b > 0)) return 0;
    return !(fabs(a - b) <= tol * (fabs(a) + 1.0));
}

/* random fill + 4-mode scales: normal / subnormal / inf-nan / signed normal */
static void fill_nasty(uint8_t *w, int n, int d) {
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    for (size_t i = 0; i < rb * (size_t)d; ++i) w[i] = (uint8_t)(rnd() & 0xFF);
    for (int r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + (size_t)r * rb + (size_t)b * PQ2_0_BLOCK_BYTES;
            uint16_t h;
            int mode = (int)(rnd() % 4);
            if (mode == 0)      h = (uint16_t)((1 + (rnd() % 29) << 10) | (rnd() & 0x3FF));
            else if (mode == 1) h = (uint16_t)(rnd() & 0x03FF);              /* subnormal */
            else if (mode == 2) h = (uint16_t)(0x7C00 | (rnd() & 0x3FF));    /* inf/nan */
            else                h = (uint16_t)((rnd() & 0x8000) | (1 + (rnd() % 29) << 10) | (rnd() & 0x3FF));
            blk[0] = (uint8_t)(h & 0xFF); blk[1] = (uint8_t)(h >> 8);
        }
}

/* scale = 1.0 everywhere (matches the pq2call/pq2scale baseline conditions) */
static uint8_t *mktiming(int n, int d) {
    const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    const size_t bytes = rb * (size_t)d;
    /* +128 B slack: the LOAD16 pair over-reads up to 48 B past the block */
    uint8_t *w = (uint8_t *)aligned_alloc(64, (bytes + 128 + 63) & ~(size_t)63);
    if (!w) { fprintf(stderr, "alloc %d x %d failed\n", n, d); exit(1); }
    uint32_t st = 12345u;
    for (size_t i = 0; i < bytes; ++i) { st = st * 1103515245u + 12345u; w[i] = (uint8_t)(st >> 16); }
    for (int r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + (size_t)r * rb + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;      /* fp16 1.0 */
        }
    return w;
}

/* the model's real folded shapes, in call order (same as pq2call.c) */
typedef struct { const char *nm; int n, d; } shp;
static const shp T[] = {
    { "ffn_gate",   5120, 17408 },
    { "ffn_up",     5120, 17408 },
    { "ffn_down",  17408,  5120 },
    { "attn_qkv",   5120, 10240 },
    { "attn_gate",  5120,  6144 },
};
#define NT ((int)(sizeof(T)/sizeof(T[0])))
#define MAXN 17408

static double measure(int use_f3, uint8_t **W, const float *x, float *y, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        for (int t = 0; t < NT; ++t) {
            const size_t rb = (size_t)(T[t].n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
            W[t][((size_t)r * 997 + (size_t)t * 37) % rb] ^= 1;  /* defeat CSE */
            if (use_f3) f3_matvec(y, x, W[t], T[t].n, T[t].d);
            else        pq2_0_matvec_knc(y, x, W[t], T[t].n, T[t].d);
            sink += y[(r * 7 + t) % (T[t].d - 1)];
        }
        const double t1 = now_s();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int nth  = argc > 1 ? atoi(argv[1]) : 244;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    omp_set_num_threads(nth);
    int gate_fail = 0;

    printf("=== pq2ab: same-binary A/B, V12 (shipped) vs F3+B2 (candidate) ===\n");
    printf("threads=%d reps=%d procs=%d\n\n", nth, reps, omp_get_num_procs());

    /* ================= GATE 1: 4 blocks x 64 alignments x 129 ============= */
    {
        uint8_t *area = (uint8_t *)aligned_alloc(64, 64 + 64 + 256);
        uint8_t blk[QK_PQ2_0 * 2 + 96];
        float x[MAXN / 4] __attribute__((aligned(64)));
        for (int i = 0; i < QK_PQ2_0; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;
        int chk12 = 0, mis12 = 0, chk3 = 0, mis3 = 0, rep1 = 0;
        for (int trial = 0; trial < 4; ++trial) {
            rng_state = 1000003u + 7919u * (uint32_t)trial;
            for (unsigned i = 0; i < sizeof blk; ++i) blk[i] = (uint8_t)(rnd() & 0xFF);
            for (int off = 0; off < 64; ++off) {
                float dr[1] __attribute__((aligned(64)));
                memset(area, 0, 64 + 64 + 256);
                memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
                rep1 += build_drow(area + off, 1, dr);
                for (int jj = 0; jj < QK_PQ2_0; ++jj) {
                    float xv[QK_PQ2_0] __attribute__((aligned(64)));
                    float xv2[QK_PQ2_0] __attribute__((aligned(64)));
                    for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
                    xv[jj] = 1.0f;
                    pq2_build_xp(xv, xv2, QK_PQ2_0);
                    const float a  = ref_row(area + off, xv, QK_PQ2_0);
                    const float b1 = v12_row(area + off, xv2, QK_PQ2_0);
                    const float b3 = f3_row(area + off, xv2, QK_PQ2_0, dr);
                    ++chk12; mis12 += bad_abs(a, b1, 1e-6f);
                    ++chk3;  mis3  += bad_abs(a, b3, 1e-6f);
                }
                {
                    float xr2[QK_PQ2_0] __attribute__((aligned(64)));
                    pq2_build_xp(x, xr2, QK_PQ2_0);
                    const float a  = ref_row(area + off, x, QK_PQ2_0);
                    const float b1 = v12_row(area + off, xr2, QK_PQ2_0);
                    const float b3 = f3_row(area + off, xr2, QK_PQ2_0, dr);
                    ++chk12; mis12 += bad_rel(a, b1, 1e-4);
                    ++chk3;  mis3  += bad_rel(a, b3, 1e-4);
                }
            }
        }
        printf("gate1 row-level: %d checks/decoder (4 blocks x 64 alignments x 129)\n", chk12);
        printf("  V12 mismatches = %d  %s\n", mis12, mis12 == 0 ? "*** EXACT ***" : "*** WRONG ***");
        printf("  F3  mismatches = %d  %s\n", mis3,  mis3  == 0 ? "*** EXACT ***" : "*** WRONG ***");
        printf("  (B2 repair path taken %d times in gate1)\n\n", rep1);
        if (mis12 || mis3) gate_fail = 1;
        free(area);
    }

    /* ================= GATE 2: forced special scales ====================== */
    {
        const int nn = 2176;                 /* 17 blocks: ODD, tail exercised */
        const int nb = nn / QK_PQ2_0;
        uint8_t *row2 = (uint8_t *)aligned_alloc(64, (size_t)nb * PQ2_0_BLOCK_BYTES + 128);
        float *x2  = (float *)aligned_alloc(64, (size_t)nn * 4);
        float *xp2 = (float *)aligned_alloc(64, (size_t)nn * 4);
        float dr[32] __attribute__((aligned(64)));
        rng_state = 777u;
        for (int i = 0; i < nb * PQ2_0_BLOCK_BYTES; ++i) row2[i] = (uint8_t)(rnd() & 0xFF);
        for (int i = 0; i < nn; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
        pq2_build_xp(x2, xp2, nn);
        /* all scales normal 1.0, then overwrite the specials */
        for (int b = 0; b < nb; ++b) { row2[b * 34] = 0x00; row2[b * 34 + 1] = 0x3C; }
        const uint16_t special[] = {
            0x0001,   /* blk 0 : min subnormal  (gather group 0 -> repair) */
            0x03FF,   /* blk 5 : max subnormal  (gather group 0 -> repair) */
            0x0400,   /* blk 9 : min normal     (fast path, edge) */
            0x7BFF,   /* blk 12: max normal     (fast path, edge) */
            0x8200,   /* blk 15: negative subnormal (group 0 -> repair) */
            0x0003,   /* blk 16: subnormal in the scalar TAIL */
        };
        const int spos[] = { 0, 5, 9, 12, 15, 16 };
        for (unsigned s = 0; s < sizeof(spos)/sizeof(spos[0]); ++s) {
            row2[spos[s] * 34]     = (uint8_t)(special[s] & 0xFF);
            row2[spos[s] * 34 + 1] = (uint8_t)(special[s] >> 8);
        }
        const int rep2 = build_drow(row2, nb, dr);
        /* bit-exact check of B2's output against the scalar converter */
        int drbad = 0;
        for (int b = 0; b < nb; ++b) {
            uint16_t h; memcpy(&h, row2 + (size_t)b * 34, 2);
            const float expect = bonsai_fp16_to_fp32(h);
            uint32_t u0, u1; memcpy(&u0, &dr[b], 4); memcpy(&u1, &expect, 4);
            if (u0 != u1) ++drbad;
        }
        const float a  = ref_row(row2, x2, nn);
        const float b1 = v12_row(row2, xp2, nn);
        const float b3 = f3_row(row2, xp2, nn, dr);
        const int m12 = bad_rel(a, b1, 1e-5), m3 = bad_rel(a, b3, 1e-5);
        printf("gate2 forced scales (17 blocks, odd; subnormal x4, min/max normal, tail):\n");
        printf("  B2 repairs = %d (expect >= 1), drow bit-exact vs scalar: %s (%d bad of %d)\n",
               rep2, drbad == 0 ? "YES" : "NO", drbad, nb);
        printf("  ref=%.9g  V12=%.9g %s   F3=%.9g %s\n\n", a, b1,
               m12 ? "*** WRONG ***" : "OK", b3, m3 ? "*** WRONG ***" : "OK");
        if (drbad || m12 || m3 || rep2 < 1) gate_fail = 1;
        free(row2); free(x2); free(xp2);
    }

    /* ============ GATE 3: multi-block, odd/even, nasty scales ============= */
    {
        const int d3 = 8;
        const size_t rb3 = (size_t)(5120 / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        uint8_t *w3 = (uint8_t *)aligned_alloc(64, rb3 * d3 + 128);
        rng_state = 4242u;
        fill_nasty(w3, 5120, d3);
        int m12 = 0, m3 = 0;
        for (int ni = 0; ni < 4; ++ni) {
            const int nn = (int[]){ 2048, 2176, 3456, 5120 }[ni];
            const int nb = nn / 128;
            float *x2  = (float *)aligned_alloc(64, (size_t)nn * 4);
            float *xp2 = (float *)aligned_alloc(64, (size_t)nn * 4);
            float dr[64] __attribute__((aligned(64)));
            for (int i = 0; i < nn; ++i) x2[i] = (float)((i % 11) - 5) * 0.125f;
            pq2_build_xp(x2, xp2, nn);
            int rep = 0;
            double w12 = 0.0, w3w = 0.0;
            for (int r = 0; r < d3; ++r) {
                const uint8_t *row = w3 + (size_t)r * rb3;
                rep += build_drow(row, nb, dr);
                const double a  = ref_row(row, x2, nn);
                const double b1 = v12_row(row, xp2, nn);
                const double b3 = f3_row(row, xp2, nn, dr);
                if (bad_rel(a, b1, 1e-5)) ++m12;
                if (bad_rel(a, b3, 1e-5)) ++m3;
                if (isfinite(a) && isfinite(b1)) {
                    const double e = fabs(a - b1) / (fabs(a) + 1.0);
                    if (e > w12) w12 = e;
                }
                if (isfinite(a) && isfinite(b3)) {
                    const double e = fabs(a - b3) / (fabs(a) + 1.0);
                    if (e > w3w) w3w = e;
                }
            }
            printf("gate3 n=%4d (%d blocks%s): 8 rows, V12 worst=%.3e mis=%d, F3 worst=%.3e mis=%d, repairs=%d\n",
                   nn, nb, (nb & 1) ? ", ODD" : "", w12, m12, w3w, m3, rep);
            free(x2); free(xp2);
        }
        printf("%s\n\n", (m12 || m3) ? "gate3 *** WRONG ***" : "gate3 OK");
        if (m12 || m3) gate_fail = 1;
        free(w3);
    }

    /* ============ GATE 4: matvec level at current thread count ============ */
    {
        const int shapes[2][2] = { { 2176, 96 }, { 5120, 48 } };
        int m12 = 0, m3 = 0, mX = 0, checks = 0;
        for (int s = 0; s < 2; ++s) {
            const int n = shapes[s][0], d = shapes[s][1];
            const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
            uint8_t *w = (uint8_t *)aligned_alloc(64, rb * (size_t)d + 128);
            float *x  = (float *)aligned_alloc(64, (size_t)n * 4);
            float *yr = (float *)aligned_alloc(64, (size_t)d * 4);
            float *y1 = (float *)aligned_alloc(64, (size_t)d * 4);
            float *y3 = (float *)aligned_alloc(64, (size_t)d * 4);
            rng_state = 999u + (uint32_t)s;
            fill_nasty(w, n, d);
            for (int i = 0; i < n; ++i) x[i] = (float)((i % 13) - 6) * 0.0625f;
            pq2_0_matvec_ref(yr, x, w, n, d);
            pq2_0_matvec_knc(y1, x, w, n, d);
            f3_matvec(y3, x, w, n, d);
            for (int i = 0; i < d; ++i) {
                ++checks;
                m12 += bad_rel(yr[i], y1[i], 1e-4);
                m3  += bad_rel(yr[i], y3[i], 1e-4);
                mX  += bad_rel(y1[i], y3[i], 1e-4);
            }
            free(w); free(x); free(yr); free(y1); free(y3);
        }
        printf("gate4 matvec-level at %d threads (%d rows, odd + even block counts):\n", nth, checks);
        printf("  V12 vs ref mismatches = %d,  F3 vs ref = %d,  F3 vs V12 = %d  %s\n\n",
               m12, m3, mX, (m12 || m3 || mX) ? "*** WRONG ***" : "OK");
        if (m12 || m3 || mX) gate_fail = 1;
    }

    /* ================= TIMING: real shape mix, both orders ================ */
    {
        long total_w = 0;
        for (int t = 0; t < NT; ++t) total_w += (long)T[t].n * T[t].d;
        float *x = (float *)aligned_alloc(64, (size_t)MAXN * 4);
        float *y = (float *)aligned_alloc(64, 17408 * 4);
        uint8_t *W[NT];
        for (int t = 0; t < NT; ++t) W[t] = mktiming(T[t].n, T[t].d);
        for (int i = 0; i < MAXN; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;

        measure(0, W, x, y, 1);      /* warm up V12 */
        measure(1, W, x, y, 1);      /* warm up F3  */
        const double v12_a = measure(0, W, x, y, reps);   /* order 1: V12 then F3 */
        const double f3_a  = measure(1, W, x, y, reps);
        const double f3_b  = measure(1, W, x, y, reps);   /* order 2: F3 then V12 */
        const double v12_b = measure(0, W, x, y, reps);

        const double g12a = total_w / v12_a / 1e9, g3a = total_w / f3_a / 1e9;
        const double g12b = total_w / v12_b / 1e9, g3b = total_w / f3_b / 1e9;
        const double g12 = g12a > g12b ? g12a : g12b;
        const double g3  = g3a  > g3b  ? g3a  : g3b;
        printf("timing: %d shapes/pass, %.3f G weights/pass, best of %d reps\n", NT, total_w / 1e9, reps);
        printf("  order V12,F3:  V12 = %8.2f G w/s   F3 = %8.2f G w/s   F3/V12 = %.3f\n", g12a, g3a, g3a / g12a);
        printf("  order F3,V12:  F3  = %8.2f G w/s   V12 = %8.2f G w/s  F3/V12 = %.3f\n", g3b, g12b, g3b / g12b);
        printf("  best each:     V12 = %8.2f G w/s   F3 = %8.2f G w/s   F3/V12 = %.3f\n\n", g12, g3, g3 / g12);
        printf("VERDICT-INPUT: gate_fail=%d  f3_wins_by=%.1f%%  (integrate iff gate_fail==0 AND win > 5%%)\n",
               gate_fail, (g3 / g12 - 1.0) * 100.0);
        for (int t = 0; t < NT; ++t) free(W[t]);
        free(x); free(y);
    }

    printf("sink=%.3f\n", (float)sink);
    return gate_fail;
}
