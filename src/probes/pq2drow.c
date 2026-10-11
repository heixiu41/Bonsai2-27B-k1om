/* pq2drow.c -- SAME-BINARY A/B: "drow precompute" for the PQ2_0 matvec.
 *
 * Shipped pq2_0_matvec_knc_xp() (src/pq2_0.h) starts EVERY ROW of its omp
 * parallel for with pq2_build_drow(): a stride-34 fp16 gather + bit-expand
 * into a thread-private stack array.  Hypothesis: that per-row, per-call
 * build costs 28-57 ms/token in-model.  This probe measures the variant
 * pq2_0_matvec_knc_xp_dpre() whose row loop instead reads a PRECOMPUTED
 * drows table (d x nblk_pad fp32 per tensor, built once at load time).
 *
 * Methodology cloned from pq2ab.c (the proven harness):
 *   - both kernels in ONE binary (cross-binary comparisons showed 20%
 *     phantom differences on this project),
 *   - real shape mix (5120x17408 x2 + 17408x5120 + 5120x10240 + 5120x6144),
 *   - best-of-N, BOTH orders (base,pre then pre,base),
 *   - a weight byte flipped every rep (defeats CSE/hoisting) -- CODE BYTES
 *     ONLY: flipping a scale byte would desync the precomputed drows table
 *     from the weights, which is precisely the update-cost caveat of this
 *     optimization (in-model weights never change after load, so it does
 *     not apply there; only the probe's anti-CSE flip must respect it),
 *   - every result feeds a printed volatile sink (defeats DCE),
 *   - xp (the permuted activation) is built ONCE per shape and is OUTSIDE
 *     the timed region, matching model.c's mm_stage() which already calls
 *     pq2_0_matvec_knc_xp() with a prebuilt xp.
 *
 * CORRECTNESS GATE (before any timing matters): y from both kernels must be
 * BIT-IDENTICAL over fill_nasty() data (random codes + normal / subnormal /
 * inf / nan fp16 scales) on all 5 real shapes, at whatever thread count the
 * binary is run with (run it at 244 and at 1).  0 mismatches required: the
 * drow CONTENT comes from the same pq2_build_drow() builder, so any nonzero
 * count is a bug in the variant, not numerics.
 *
 * drows layout: row i's scales at drows + i * nblk_pad, with
 * nblk_pad = round_up(n/128, 16) so every row base is 64-byte aligned
 * (pq2_build_drow's fast path uses _mm512_store_ps at drow + 16k).
 *
 * KNC traps respected: no loadu/storeu, gathers via the proven
 * _mm512_i32extgather_epi32 UINT16 upconvert, _mm512_load_ps only on
 * 64-byte-aligned buffers, kmp_set_stacksize_s before the 244-thread regions.
 *
 * Build (Windows host, icc 2017 for k1om):
 *   $icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
 *   $env:MIC_SYSROOT = 'C:\Program Files\Intel\MPSS'
 *   & $icc -mmic -O2 -std=c11 -qopenmp -Isrc src\probes\pq2drow.c -o build\pq2drow -lm
 * Run on the card:
 *   scp -O -c chacha20-poly1305@openssh.com -i $key build\pq2drow root@192.168.1.100:/tmp/
 *   ssh ... root@192.168.1.100 "export LD_LIBRARY_PATH=/tmp; /tmp/pq2drow 244 5"
 *   ssh ... root@192.168.1.100 "export LD_LIBRARY_PATH=/tmp; /tmp/pq2drow 1 5"
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

/* ======================= THE VARIANT UNDER TEST ==========================
 * Bit-for-bit the shipped pq2_0_matvec_knc_xp() row body, but the row loop
 * reads its scales from the precomputed table instead of building drow on
 * the stack.  drows row stride is nblk_pad (see header). */
static void pq2_0_matvec_knc_xp_dpre(float *y, const float *xp,
                                     const uint8_t *w, int n, int d,
                                     const float *drows) {
    const int nn = n;                       /* multiple of QK_PQ2_0 */
    const size_t rb = ((size_t)n + QK_PQ2_0 - 1) / QK_PQ2_0 * PQ2_0_BLOCK_BYTES;
    const __m512 l0b = pq2_lut_a(), l1b = pq2_lut_b();
    const int nblk     = nn / QK_PQ2_0;
    const int nblk_pad = (nblk + 15) & ~15;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        const uint8_t *row = w + (size_t)i * rb;
        __m512 s = _mm512_setzero_ps();
        const uint8_t *blk = row;
        const float *x0 = xp;
        const float *dp = drows + (size_t)i * nblk_pad;
        int b = nblk;
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
        y[i] = _mm512_reduce_add_ps(s);
    }
}

/* One-time (per model load) builder: every row's drow, omp-parallel over
 * rows, same builder the hot kernel uses today. */
static void pq2_build_all_drows(const uint8_t *w, int n, int d, float *drows) {
    const int nblk     = n / QK_PQ2_0;
    const int nblk_pad = (nblk + 15) & ~15;
    const size_t rb    = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i)
        pq2_build_drow(w + (size_t)i * rb, nblk, drows + (size_t)i * nblk_pad);
}

/* ---------------- data generators (verbatim from pq2ab.c) --------------- */
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

/* the model's real folded shapes, in call order (same as pq2ab.c/pq2call.c) */
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

/* one timed pass over the mix (t == -1) or a single shape; best of reps.
 * The anti-CSE flip touches ONLY code bytes (blk*34 + 2 + ...): a scale-byte
 * flip would desync the precomputed drows table and the comparison would
 * measure a stale-cache artifact, not the optimization. */
static double measure(int use_pre, int t, uint8_t **W, float **DR,
                      float **XP, float *y, int reps) {
    double best = 1e30;
    const int lo = t < 0 ? 0 : t;
    const int hi = t < 0 ? NT : t + 1;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        for (int s = lo; s < hi; ++s) {
            const int nblk = T[s].n / QK_PQ2_0;
            const int nb = (r * 997 + s * 37) % nblk;
            W[s][(size_t)nb * PQ2_0_BLOCK_BYTES + 2 + (r & 31)] ^= 1;  /* code byte */
            if (use_pre) pq2_0_matvec_knc_xp_dpre(y, XP[s], W[s], T[s].n, T[s].d, DR[s]);
            else         pq2_0_matvec_knc_xp(y, XP[s], W[s], T[s].n, T[s].d);
            sink += y[(r * 7 + s) % (T[s].d - 1)];
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
    kmp_set_stacksize_s(4 * 1024 * 1024);
    omp_set_num_threads(nth);
    int gate_fail = 0;

    printf("=== pq2drow: same-binary A/B, drow-per-row (shipped) vs drow-precomputed ===\n");
    printf("threads=%d reps=%d procs=%d\n\n", nth, reps, omp_get_num_procs());

    /* ============ GATE: bit-identical y on all 5 real shapes =============
     * fill_nasty covers random codes + normal / subnormal / inf / nan fp16
     * scales.  NaN/inf outputs are fine: both kernels run the same FMA
     * sequence on the same scale bits, so even NaN payloads must match. */
    {
        int total_mis = 0;
        for (int s = 0; s < NT; ++s) {
            const int n = T[s].n, d = T[s].d;
            const int nblk = n / QK_PQ2_0;
            const int nblk_pad = (nblk + 15) & ~15;
            const size_t rb = (size_t)nblk * PQ2_0_BLOCK_BYTES;
            uint8_t *w  = (uint8_t *)aligned_alloc(64, (rb * (size_t)d + 128 + 63) & ~(size_t)63);
            float   *x  = (float *)aligned_alloc(64, (size_t)n * 4);
            float   *xp = (float *)aligned_alloc(64, (size_t)n * 4);
            float   *dr = (float *)aligned_alloc(64, (size_t)nblk_pad * (size_t)d * 4);
            float   *y1 = (float *)aligned_alloc(64, (size_t)d * 4);
            float   *y2 = (float *)aligned_alloc(64, (size_t)d * 4);
            if (!w || !x || !xp || !dr || !y1 || !y2) { fprintf(stderr, "gate alloc failed\n"); return 2; }
            rng_state = 999u + (uint32_t)s;
            fill_nasty(w, n, d);
            for (int i = 0; i < n; ++i) x[i] = (float)((i % 13) - 6) * 0.0625f;
            pq2_build_xp(x, xp, n);
            pq2_build_all_drows(w, n, d, dr);
            pq2_0_matvec_knc_xp(y1, xp, w, n, d);
            pq2_0_matvec_knc_xp_dpre(y2, xp, w, n, d, dr);
            int mis = 0, first = -1;
            for (int i = 0; i < d; ++i) {
                uint32_t u1, u2;
                memcpy(&u1, y1 + i, 4); memcpy(&u2, y2 + i, 4);
                if (u1 != u2) { if (first < 0) first = i; ++mis; }
            }
            printf("gate %9s %5dx%-5d: bit mismatches = %d of %d %s\n", T[s].nm, n, d,
                   mis, d, mis == 0 ? "*** IDENTICAL ***" : "*** WRONG ***");
            if (mis && first >= 0) {
                uint32_t u1, u2;
                memcpy(&u1, y1 + first, 4); memcpy(&u2, y2 + first, 4);
                printf("   first mismatch row %d: base=%08x pre=%08x\n", first, u1, u2);
            }
            total_mis += mis;
            free(w); free(x); free(xp); free(dr); free(y1); free(y2);
        }
        printf("gate total at %d threads: %d mismatches  %s\n\n", nth,
               total_mis, total_mis == 0 ? "OK" : "*** WRONG ***");
        if (total_mis) gate_fail = 1;
    }

    /* ================= setup for timing (scale = 1.0 data) =============== */
    long total_w = 0;
    for (int t = 0; t < NT; ++t) total_w += (long)T[t].n * T[t].d;
    float *x = (float *)aligned_alloc(64, (size_t)MAXN * 4);
    float *y = (float *)aligned_alloc(64, 17408 * 4);
    uint8_t *W[NT]; float *DR[NT]; float *XP[NT];
    size_t dr_bytes_mix = 0;
    for (int t = 0; t < NT; ++t) {
        const int nblk_pad = ((T[t].n / QK_PQ2_0) + 15) & ~15;
        W[t]  = mktiming(T[t].n, T[t].d);
        DR[t] = (float *)aligned_alloc(64, (size_t)nblk_pad * (size_t)T[t].d * 4);
        XP[t] = (float *)aligned_alloc(64, (size_t)T[t].n * 4);
        if (!DR[t] || !XP[t]) { fprintf(stderr, "timing alloc failed\n"); return 2; }
        dr_bytes_mix += (size_t)nblk_pad * (size_t)T[t].d * 4;
    }
    for (int i = 0; i < MAXN; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;
    for (int t = 0; t < NT; ++t) {
        pq2_build_xp(x, XP[t], T[t].n);
        pq2_build_all_drows(W[t], T[t].n, T[t].d, DR[t]);
    }

    /* ============ ONE-TIME COST: build ALL drows for the mix ============= */
    {
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            for (int t = 0; t < NT; ++t) {
                pq2_build_all_drows(W[t], T[t].n, T[t].d, DR[t]);
                sink += DR[t][(size_t)(r * 13) % ((size_t)T[t].d)];
            }
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        long total_blk = 0;
        for (int t = 0; t < NT; ++t) total_blk += (long)(T[t].n / QK_PQ2_0) * T[t].d;
        printf("one-time drow build (all 5 shapes, %d threads, best of %d):\n", nth, reps);
        printf("  %.2f ms for %.2f M blocks (%.1f ns/block), mix drows = %.1f MiB\n\n",
               best * 1e3, total_blk / 1e6, best * 1e9 / (double)total_blk,
               dr_bytes_mix / 1048576.0);
    }

    /* ================= TIMING: mix, both orders ========================== */
    {
        measure(0, -1, W, DR, XP, y, 1);      /* warm up base */
        measure(1, -1, W, DR, XP, y, 1);      /* warm up pre  */
        const double base_a = measure(0, -1, W, DR, XP, y, reps);   /* order 1: base then pre */
        const double pre_a  = measure(1, -1, W, DR, XP, y, reps);
        const double pre_b  = measure(1, -1, W, DR, XP, y, reps);   /* order 2: pre then base */
        const double base_b = measure(0, -1, W, DR, XP, y, reps);

        const double gba = total_w / base_a / 1e9, gpa = total_w / pre_a / 1e9;
        const double gbb = total_w / base_b / 1e9, gpb = total_w / pre_b / 1e9;
        const double gb = gba > gbb ? gba : gbb;
        const double gp = gpa > gpb ? gpa : gpb;
        printf("timing (mix): %d shapes/pass, %.3f G weights/pass, best of %d reps\n",
               NT, total_w / 1e9, reps);
        printf("  order base,pre:  base = %8.2f G w/s   pre = %8.2f G w/s   pre/base = %.3f\n",
               gba, gpa, gpa / gba);
        printf("  order pre,base:  pre  = %8.2f G w/s   base = %8.2f G w/s  pre/base = %.3f\n",
               gpb, gbb, gpb / gbb);
        printf("  best each:       base = %8.2f G w/s   pre  = %8.2f G w/s  pre/base = %.3f\n\n",
               gb, gp, gp / gb);

        /* per-shape numbers, both orders per shape */
        printf("per-shape (best of both orders, best of %d reps):\n", reps);
        printf("  %-9s %13s %13s %13s %9s\n", "shape", "n x d", "base G w/s", "pre G w/s", "pre/base");
        for (int t = 0; t < NT; ++t) {
            const long sw = (long)T[t].n * T[t].d;
            const double b1 = measure(0, t, W, DR, XP, y, reps);
            const double p1 = measure(1, t, W, DR, XP, y, reps);
            const double p2 = measure(1, t, W, DR, XP, y, reps);
            const double b2 = measure(0, t, W, DR, XP, y, reps);
            const double bs = b1 < b2 ? b2 : b1, ps = p1 < p2 ? p2 : p1;
            char dim[32]; snprintf(dim, sizeof dim, "%dx%d", T[t].n, T[t].d);
            printf("  %-9s %13s %13.2f %13.2f %9.3f\n", T[t].nm, dim,
                   sw / bs / 1e9, sw / ps / 1e9, bs / ps);
        }
        printf("\n");

        /* ================= token economics ============================== */
        const double TW = 26.87e9;        /* model weights touched per token */
        const double tok_base = TW / (gb * 1e9), tok_pre = TW / (gp * 1e9);
        const double drow_model_bytes = TW / QK_PQ2_0 * 4.0;
        printf("token economics (26.87 G weights/token, at this thread count):\n");
        printf("  matvec time/token:  base = %7.1f ms   pre = %7.1f ms   saved = %6.1f ms\n",
               tok_base * 1e3, tok_pre * 1e3, (tok_base - tok_pre) * 1e3);
        printf("  full-model drow memory: 26.87e9/128 * 4 B = %.0f MB (%.1f MiB)\n\n",
               drow_model_bytes / 1e6, drow_model_bytes / 1048576.0);
        printf("VERDICT-INPUT: gate_fail=%d  pre_wins_by=%.1f%%  saves %.1f ms/token  costs %.0f MB\n",
               gate_fail, (gp / gb - 1.0) * 100.0, (tok_base - tok_pre) * 1e3,
               drow_model_bytes / 1e6);
    }

    for (int t = 0; t < NT; ++t) { free(W[t]); free(DR[t]); free(XP[t]); }
    free(x); free(y);
    printf("sink=%.3f\n", (float)sink);
    return gate_fail;
}
