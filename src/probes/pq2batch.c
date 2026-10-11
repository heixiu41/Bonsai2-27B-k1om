/* Does batching actually amortise the decode? A decisive microbenchmark.
 *
 * WHY. The decoder is 39% of a token (246 ms of 628 ms), and it is COMPUTE
 * bound: 26.87e9 weights/token x 40 ops/128 = 8.4 G vector ops, at 102 G
 * weights/s effective. Everything else has been ruled out by measurement:
 *   - working set / cache   : 6 GB vs 23 MB costs 3.6%
 *   - parallel region cost  : 401 x 34 us = 13.6 ms/token (5%)
 *   - thread idle/imbalance : 244/244 threads report, imbalance 1.02x
 *   - cheap instructions    : none (all 512-bit ops cost the same)
 *   - integer/FP dual issue : none (one shared pipe)
 * So the only levers left are (a) fewer ops per weight and (b) pay the decode
 * once and reuse it for several tokens.
 *
 * This measures (b) directly, in isolation, before anyone refactors a 1100-line
 * forward pass on a guess.
 *
 * THE COMPARISON. For one 128-weight block and B activations:
 *   UNBATCHED: for each of the B tokens, decode the block (4 LOAD16 + 8 permute
 *              + 8 shift + 8 mul) then do 8 FMAs -> decode paid B times.
 *   BATCHED  : decode the block ONCE into 8 float vectors, then for each token
 *              do 8 FMAs against that token's x -> decode paid once.
 * The FMA count is identical (B x 8); only the decode is amortised. So the
 * speedup measures exactly what batching buys.
 *
 * Traps avoided: results go into a printed volatile sink; each repetition uses
 * a different weight byte and a different x so nothing is hoisted; buffers are
 * allocated once.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#include "pq2_0.h"

static volatile float sink;
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* decode one 128-weight block into 8 float vectors (the shipped sequence) */
static inline void decode_block(const uint8_t *blk, __m512 g[8]) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const __m512i shf = PQ2_IMCI_SHIFT();
    const __m512 vd  = _mm512_set1_ps(d);
    const __m512i c0 = PQ2_IMCI_LOAD16(qs);
    const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
    #define G(m, c) do {                                                    \
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c);       \
        const __m512i v = _mm512_srlv_epi32(b, shf);                        \
        g[m] = _mm512_mul_ps(                                                \
            _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);       \
    } while (0)
    G(0,c0); G(1,c0); G(2,c0); G(3,c0);
    G(4,c1); G(5,c1); G(6,c1); G(7,c1);
    #undef G
}

int main(int argc, char **argv) {
    const int nblocks = argc > 1 ? atoi(argv[1]) : 4096;
    const int nth     = argc > 2 ? atoi(argv[2]) : 1;
    omp_set_num_threads(nth);

    const size_t bytes = (size_t)nblocks * PQ2_0_BLOCK_BYTES + 256;
    uint8_t *w = (uint8_t *)malloc(bytes);
    if (!w) return 1;
    uint32_t st = 99u;
    for (size_t i = 0; i < bytes; ++i) { st = st*1103515245u+12345u; w[i] = (uint8_t)(st>>16); }
    for (int b = 0; b < nblocks; ++b) {
        uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
        blk[0] = 0x00; blk[1] = 0x3C;                 /* fp16 1.0 */
    }

    /* B activation vectors of 128 floats each */
    #define BMAX 16
    float *xs = (float *)aligned_alloc(64, (size_t)BMAX * 128 * sizeof(float));
    if (!xs) return 1;
    for (int i = 0; i < BMAX * 128; ++i) xs[i] = (float)((i % 7) - 3) * 0.25f;

    printf("=== batched vs unbatched decode: %d blocks, %d threads ===\n\n", nblocks, nth);
    printf("  %6s %14s %14s %12s %12s\n",
           "B", "unbatched M/s", "batched M/s", "speedup", "ops/128 eff");

    const int Bs[] = { 1, 2, 4, 8, 16 };
    for (unsigned bi = 0; bi < sizeof(Bs)/sizeof(Bs[0]); ++bi) {
        const int B = Bs[bi];
        if (B > BMAX) break;
        double bestU = 1e30, bestB = 1e30;

        for (int rep = 0; rep < 3; ++rep) {
            /* ---- unbatched: decode again for every token ---- */
            w[(size_t)rep] ^= 1;
            double t0 = now_s();
            #pragma omp parallel for schedule(static) reduction(+:sink)
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
                float acc = 0;
                for (int t = 0; t < B; ++t) {
                    __m512 g[8];
                    decode_block(blk, g);                       /* decoded B times */
                    const float *x = xs + (size_t)t * 128;
                    __m512 s = _mm512_setzero_ps();
                    for (int m = 0; m < 8; ++m)
                        s = _mm512_fmadd_ps(g[m], _mm512_load_ps(x + 16*m), s);
                    acc += _mm512_reduce_add_ps(s);
                }
                sink += acc;
            }
            double t1 = now_s();
            if (t1 - t0 < bestU) bestU = t1 - t0;

            /* ---- batched: decode once, reuse across tokens ---- */
            t0 = now_s();
            #pragma omp parallel for schedule(static) reduction(+:sink)
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t *blk = w + (size_t)b * PQ2_0_BLOCK_BYTES;
                __m512 g[8];
                decode_block(blk, g);                           /* decoded ONCE */
                float acc = 0;
                for (int t = 0; t < B; ++t) {
                    const float *x = xs + (size_t)t * 128;
                    __m512 s = _mm512_setzero_ps();
                    for (int m = 0; m < 8; ++m)
                        s = _mm512_fmadd_ps(g[m], _mm512_load_ps(x + 16*m), s);
                    acc += _mm512_reduce_add_ps(s);
                }
                sink += acc;
            }
            t1 = now_s();
            if (t1 - t0 < bestB) bestB = t1 - t0;
        }

        const double weights = (double)nblocks * 128 * B;   /* weight-uses */
        const double mu = weights / bestU / 1e6;
        const double mb = weights / bestB / 1e6;
        printf("  %6d %14.1f %14.1f %11.2fx %12s\n", B, mu, mb, mb/mu, "");
    }

    printf("\nsink %.3f\n", (float)sink);
    printf("\n'batched/unbatched' is the pure win from paying the decode once.\n");
    printf("If it is large, restructuring the forward pass to batch tokens is worth it.\n");
    free(w); free(xs);
    return 0;
}
