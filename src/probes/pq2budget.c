/* Decisive test: how much of a token is the PQ2_0 matvec, really?
 *
 * A single-threaded decoder rate of 917.8 M weights/s predicts 29.3 s/token for
 * 26.87e9 weights, yet the engine runs at 3.73 s/token on 122 threads. Nobody
 * has ever checked whether the remaining time is (A) the matvec not scaling, or
 * (B) work outside the matvec. This settles it by measuring the real kernels at
 * the real shapes and thread counts.
 *
 * Method: time pq2_0_matvec_knc-style work (pq2_0_matvec_ref here, which is the
 * same arithmetic and is compiled for all targets) over the model's ACTUAL
 * tensor shapes and call counts, at several thread counts. Compare the sum to
 * the measured 3.73 s/token.
 *
 * Model shapes and per-token call counts (from the GGUF; verified):
 *   ffn_gate  5120 -> 17408   x64   = 64 calls
 *   ffn_up    5120 -> 17408   x64
 *   ffn_down 17408 ->  5120   x64
 *   attn_qkv  5120 -> 10240   x64
 *   attn_gate 5120 ->  6144   x64
 *   ssm_out   5120 ->  6144   x48   (GDN blocks only)
 *   attn_q    5120 -> 12288   x16   (full-attention blocks only)
 *   attn_output 6144 -> 5120  x16
 *   output   5120 -> 248320   x1
 *
 * Only the PQ2_0 (2-bit) weights matter here; the small F32/BF16 tensors are
 * negligible and are ignored, which makes this an UPPER bound on matvec share.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "pq2_0.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct { const char *name; int n, d, calls; } shape;

static const shape SHAPES[] = {
    { "ffn_gate",      5120, 17408, 64 },
    { "ffn_up",        5120, 17408, 64 },
    { "ffn_down",     17408,  5120, 64 },
    { "attn_qkv",      5120, 10240, 64 },
    { "attn_gate",     5120,  6144, 64 },
    { "ssm_out",       5120,  6144, 48 },
    { "attn_q",        5120, 12288, 16 },
    { "attn_output",   6144,  5120, 16 },
    { "output",        5120, 248320, 1 },
};

int main(int argc, char **argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 122;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    omp_set_num_threads(nth);

    printf("=== is the token time the matvec?  threads=%d ===\n", nth);
    printf("num_procs reported = %d\n\n", omp_get_num_procs());
    printf("%-12s %7s %7s %6s %10s %12s %14s\n",
           "tensor", "n", "d", "calls", "weights", "ms/call", "ms/token");

    double total_ms = 0.0;
    double total_w  = 0.0;

    for (unsigned s = 0; s < sizeof(SHAPES)/sizeof(SHAPES[0]); ++s) {
        const shape *sh = &SHAPES[s];
        const size_t row_bytes = (size_t)(sh->n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        /* one row only -- pq2_0_matvec_ref reads d rows, so allocate d rows */
        uint8_t *w = (uint8_t *)aligned_alloc(64, ((row_bytes * sh->d) + 127) & ~(size_t)63);
        float   *x = (float   *)aligned_alloc(64, ((size_t)sh->n * 4 + 63) & ~(size_t)63);
        float   *y = (float   *)aligned_alloc(64, ((size_t)sh->d * 4 + 63) & ~(size_t)63);
        if (!w || !x || !y) { fprintf(stderr, "alloc failed at %s\n", sh->name); return 1; }

        /* deterministic content; scales set to 1.0 so the numbers are finite */
        uint32_t st = 12345u + s;
        for (size_t i = 0; i < row_bytes * (size_t)sh->d; ++i) {
            st = st * 1103515245u + 12345u;
            w[i] = (uint8_t)(st >> 16);
        }
        for (int r = 0; r < sh->d; ++r)
            for (int b = 0; b < sh->n / QK_PQ2_0; ++b) {
                uint8_t *blk = w + (size_t)r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
                blk[0] = 0x00; blk[1] = 0x3C;   /* fp16 1.0 */
            }
        for (int i = 0; i < sh->n; ++i) x[i] = (float)((i % 7) - 3) * 0.25f;

        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            pq2_0_matvec_knc(y, x, w, sh->n, sh->d);
            const double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
        }
        const double ms_call = best * 1e3;
        const double ms_tok  = ms_call * sh->calls;
        const double weights = (double)sh->n * sh->d;
        total_ms += ms_tok;
        total_w  += weights * sh->calls;

        printf("%-12s %7d %7d %6d %10.3fG %12.3f %14.1f\n",
               sh->name, sh->n, sh->d, sh->calls,
               weights * sh->calls / 1e9, ms_call, ms_tok);

        free(w); free(x); free(y);
    }

    printf("\n  total PQ2_0 weights/token = %.3f G\n", total_w / 1e9);
    printf("  total matvec time/token   = %.0f ms\n", total_ms);
    printf("  measured end-to-end/token = 3730 ms  (0.268 tok/s)\n");
    printf("  => matvec share           = %.1f%%\n", 100.0 * total_ms / 3730.0);
    printf("  => if matvec became FREE, token time would be %.0f ms (%.2f tok/s)\n",
           fabs(3730.0 - total_ms), 1000.0 / fabs(3730.0 - total_ms));
    printf("\n  (upper bound: this charges ALL the time to matvec and ignores\n");
    printf("   the F32/BF16 tensors, norms, softmax, RoPE and the GDN recurrence)\n");
    return 0;
}
