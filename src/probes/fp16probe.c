/*
 * fp16probe.c -- exhaustive check of bonsai_fp16_to_fp32() against the
 * platform's own half->float conversion (via numpy-generated table), plus the
 * PQ2_0 code map and the fwht/normalisation of hadamard_forward().
 *
 * Emits one line per input so the host driver can diff against numpy.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "pq2_0.h"

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "fp16") == 0) {
        for (uint32_t h = 0; h < 65536u; ++h) {
            union { float f; uint32_t u; } v;
            v.f = bonsai_fp16_to_fp32((uint16_t)h);
            printf("%u %u\n", h, v.u);        /* compare bit patterns */
        }
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "hadamard") == 0) {
        /* x' = H (s x) with H = WHT/sqrt(block); check against the explicit
         * matrix for block = 8 and a known sign vector. */
        const int block = 8, n = 8;
        float x[8] = { 1, -2, 3, -4, 5, -6, 7, -8 };
        float s[8] = { 1, -1, 1, 1, -1, -1, 1, -1 };
        float raw[8], got[8];
        for (int i = 0; i < n; ++i) raw[i] = x[i];
        hadamard_forward(raw, s, n, block, 0);
        for (int r = 0; r < n; ++r) {
            float acc = 0.0f;
            for (int c = 0; c < n; ++c) {
                const int par = __builtin_popcount((unsigned)(r & c)) & 1;
                acc += (par ? -1.0f : 1.0f) * s[c] * x[c];
            }
            got[r] = acc / sqrtf((float)block);
        }
        float e = 0.0f;
        for (int i = 0; i < n; ++i) { float d = fabsf(got[i] - raw[i]); if (d > e) e = d; }
        printf("hadamard max_err=%.3e\n", e);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "dequant") == 0) {
        /* code -> value mapping for all 256 byte values in every byte slot */
        int bad = 0;
        for (int slot = 0; slot < 32; ++slot) {
            for (int byte = 0; byte < 256; ++byte) {
                uint8_t blk[PQ2_0_BLOCK_BYTES];
                memset(blk, 0, sizeof(blk));
                blk[0] = 0x00; blk[1] = 0x3C;      /* d = 1.0 */
                blk[2 + slot] = (uint8_t)byte;
                float out[QK_PQ2_0];
                pq2_0_dequant_block(blk, out);
                for (int k = 0; k < 4; ++k) {
                    const int q = (byte >> (2 * k)) & 3;
                    const float want = (float)(q - 1);
                    if (out[4 * slot + k] != want) ++bad;
                }
            }
        }
        printf("dequant code-map mismatches=%d (over 32*256*4 element checks)\n", bad);
        return 0;
    }
    fprintf(stderr, "usage: %s fp16|hadamard|dequant\n", argv[0]);
    return 2;
}
