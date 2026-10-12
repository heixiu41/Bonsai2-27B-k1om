/* pq2repack.c -- codes-only repack: same-process A/B of the 34-byte and
 * 32-byte (packed) F3 kernels, on the five real shapes.
 *
 * Method (same as src/probes/pq2ab.c):
 *   - both orders timed alternately inside ONE process, best-of-5
 *   - a code byte is flipped between reps so the compiler cannot hoist the
 *     kernel out of the loop (anti-CSE), and the results feed a volatile sink
 *   - 1 and 244 threads
 *   - every row's output compared bit-for-bit between the two layouts
 *
 * The working set is deliberately larger than the aggregate L2 (244 x 512 KB)
 * so the 244-thread numbers stream from DRAM like the real model does; the
 * 1-thread numbers are where the instruction-count difference shows.
 *
 * Build: icc -mmic -O2 -std=c11 -qopenmp -I../ pq2repack.c -o pq2repack
 * Run:   ./pq2repack [copies_mb] [reps]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#include "pq2_0.h"

struct shape { int n, d; const char *name; };

static const struct shape SH[] = {
    {  5120, 17408, "5120x17408 (ffn gate/up)" },
    { 17408,  5120, "17408x5120 (ffn down)"    },
    {  5120, 10240, "5120x10240 (gdn qkv)"     },
    {  5120,  6144, "5120x6144  (gdn z/value)" },
    {  5120,  2048, "5120x2048  (attn k/v)"    },
};
#define NSH (int)(sizeof SH / sizeof SH[0])

static volatile float sink;
static uint32_t rng = 987654321u;
static uint32_t rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 9; }

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const size_t target = (argc > 1 ? (size_t)atoi(argv[1]) : 256) << 20;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const int maxth = omp_get_max_threads();

    printf("threads available %d, target working set %zu MiB, best of %d\n\n",
           maxth, target >> 20, reps);
    printf("%-26s %4s %10s %10s %8s %6s %10s\n", "shape", "thr",
           "34B G w/s", "32B G w/s", "ratio", "copies", "bit-equal");

    for (int s = 0; s < NSH; ++s) {
        const int n = SH[s].n, d = SH[s].d;
        const int nblk = n / QK_PQ2_0;
        const int nblk_pad = (nblk + 15) & ~15;
        const size_t raw_rb = (size_t)nblk * PQ2_0_BLOCK_BYTES;      /* 34 */
        const size_t pk_rb  = (size_t)nblk * PQ2_0_PACKED_BLOCK_BYTES; /* 32 */

        size_t copies = target / ((size_t)d * raw_rb);
        if (copies < 1) copies = 1;
        if (copies > 4096) copies = 4096;
        const size_t rows = copies * (size_t)d;
        const size_t raw_bytes = copies * (size_t)d * raw_rb;

        uint8_t *raw = aligned_alloc(64, raw_bytes + 128);   /* +128: slack for
                                                             * the unpack over-read */
        uint8_t *pk  = aligned_alloc(64, rows * pk_rb + 128);
        float   *dr  = aligned_alloc(64, rows * (size_t)nblk_pad * sizeof(float));
        float   *xp  = aligned_alloc(64, (size_t)n * sizeof(float));
        float   *y0  = aligned_alloc(64, rows * sizeof(float));
        float   *y1  = aligned_alloc(64, rows * sizeof(float));
        if (!raw || !pk || !dr || !xp || !y0 || !y1) { perror("alloc"); return 1; }
        memset(raw, 0, raw_bytes + 128);
        memset(pk, 0, rows * pk_rb + 128);

        /* block content: a valid fp16 scale plus 32 random code bytes */
        for (size_t i = 0; i < rows; ++i) {
            uint8_t *r = raw + i * raw_rb;
            for (int b = 0; b < nblk; ++b, r += PQ2_0_BLOCK_BYTES) {
                /* normal fp16 in [2^-14, 2^-1); ~1% subnormal to exercise the
                 * B2 repair path */
                const uint16_t h = (rnd() % 100 == 0)
                                 ? (uint16_t)(1 + rnd() % 0x3FF)
                                 : (uint16_t)(0x2000 + rnd() % 0x1C00);
                r[0] = (uint8_t)(h & 0xFF);
                r[1] = (uint8_t)(h >> 8);
                for (int k = 2; k < PQ2_0_BLOCK_BYTES; ++k)
                    r[k] = (uint8_t)rnd();
            }
        }
        /* packed copy: codes only, 32 bytes per block, strides 32 */
        for (size_t i = 0; i < rows; ++i) {
            const uint8_t *r = raw + i * raw_rb;
            uint8_t *q = pk + i * pk_rb;
            for (int b = 0; b < nblk; ++b) {
                memcpy(q, r + 2, 32);
                r += PQ2_0_BLOCK_BYTES;
                q += PQ2_0_PACKED_BLOCK_BYTES;
            }
        }
        pq2_build_drows_mt(raw, raw_rb, nblk, (int)rows, dr);
        for (int i = 0; i < n; ++i)
            xp[i] = (float)((int)(rnd() % 2001) - 1000) * 0.0004f;

        const uint64_t wts = (uint64_t)n * (uint64_t)rows;

        /* ---- 1. bit-exactness on PRISTINE data --------------------------
         * The two kernels perform the identical sequence of FMAs in the same
         * order; only the widening load differs, and pq2repack proved those
         * lane-identical.  So this must be an exact match, not "close". */
        pq2_0_matvec_knc_xp_dpre(y0, xp, raw, n, (int)rows, dr, 0);
        pq2_0_matvec_knc_xp_packed(y1, xp, pk, n, (int)rows, dr, 0);
        int biteq = 1;
        size_t firstbad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            if (memcmp(&y0[i], &y1[i], 4) != 0) {
                if (biteq) firstbad = i;
                biteq = 0;
                const double a = y0[i], b = y1[i];
                const double den = fabs(a) > 1e-30 ? fabs(a) : 1e-30;
                const double r = fabs(a - b) / den;
                if (r > maxrel) maxrel = r;
            }
        }

        /* ---- 2. timing, both orders alternating ------------------------ */
        double best_dpre[2] = { 1e30, 1e30 }, best_pk[2] = { 1e30, 1e30 };
        for (int t = 0; t < 2; ++t) {
            const int thr = (t == 0) ? 1 : maxth;
            omp_set_num_threads(thr);
            double b0 = 1e30, b1 = 1e30;
            for (int r = 0; r < reps; ++r) {
                /* anti-CSE: flip the SAME logical code in both layouts */
                const size_t blk = (size_t)(rnd() % (uint32_t)(rows * (size_t)nblk));
                const size_t row = blk / (size_t)nblk, b = blk % (size_t)nblk;
                const int k = (int)(rnd() % 32);
                raw[row * raw_rb + b * PQ2_0_BLOCK_BYTES + 2 + k] ^= 1;
                pk[row * pk_rb + b * PQ2_0_PACKED_BLOCK_BYTES + k] ^= 1;

                double t0, t1, t2;
                if (r & 1) {                       /* packed first */
                    t0 = now_s();
                    pq2_0_matvec_knc_xp_packed(y1, xp, pk, n, (int)rows, dr, 0);
                    t1 = now_s();
                    pq2_0_matvec_knc_xp_dpre(y0, xp, raw, n, (int)rows, dr, 0);
                    t2 = now_s();
                    best_pk[t] = (t1 - t0) < best_pk[t] ? (t1 - t0) : best_pk[t];
                    best_dpre[t] = (t2 - t1) < best_dpre[t] ? (t2 - t1) : best_dpre[t];
                } else {                           /* dpre first */
                    t0 = now_s();
                    pq2_0_matvec_knc_xp_dpre(y0, xp, raw, n, (int)rows, dr, 0);
                    t1 = now_s();
                    pq2_0_matvec_knc_xp_packed(y1, xp, pk, n, (int)rows, dr, 0);
                    t2 = now_s();
                    best_dpre[t] = (t1 - t0) < best_dpre[t] ? (t1 - t0) : best_dpre[t];
                    best_pk[t] = (t2 - t1) < best_pk[t] ? (t2 - t1) : best_pk[t];
                }
                sink += y0[0] + y1[0] * 1e-30f;
            }
            printf("%-26s %4d %10.1f %10.1f %8.3f %6zu %10s",
                   SH[s].name, thr, wts / best_dpre[t] / 1e9,
                   wts / best_pk[t] / 1e9, best_dpre[t] / best_pk[t],
                   copies, biteq ? "yes" : "NO");
            if (!biteq)
                printf("   first bad row %zu, max rel %.2e", firstbad, maxrel);
            printf("\n");
            fflush(stdout);
        }
        free(raw); free(pk); free(dr); free(xp); free(y0); free(y1);
    }
    printf("\nsink %.6g\n", (double)sink);
    return 0;
}
