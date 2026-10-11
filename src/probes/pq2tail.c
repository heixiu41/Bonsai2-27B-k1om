/*
 * pq2tail.c -- review probe (k1om): IMCI matvec vs scalar reference across
 * shapes, including n that is NOT a multiple of 128 (the tail path), plus a
 * decoder sweep over adversarial byte patterns and a check that poison in the
 * slack region past a block cannot change the decoded weights.
 *
 *   icc -mmic -O2 -std=c11 -qopenmp pq2tail.c -o pq2tail -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "pq2_0.h"

/* ---- deterministic PRNG so the run is reproducible ---- */
static uint32_t rs = 12345u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* 64-byte aligned static scratch: _mm512_load_ps/_mm512_store_ps fault
 * otherwise (measured: SIGSEGV, see pq2read.c section C). */
#define XCAP (17408 + 64)
static float xbuf[XCAP] __attribute__((aligned(64)));

/* bit-exact comparison so that NaN == NaN (a != b is true for two NaNs) */
static int same_bits(float x, float y) {
    uint32_t a, b;
    memcpy(&a, &x, 4); memcpy(&b, &y, 4);
    return a == b;
}

static int reported = 0;
static void report_mismatch(const char *tag, const uint8_t *blk, int lane,
                            float a, float b) {
    if (reported++ >= 8) return;
    printf("    MISMATCH %s lane=%d  scalar=%.9g imci=%.9g   blk: d=%02x%02x qs[%d]=%02x\n",
           tag, lane, (double)a, (double)b, blk[0], blk[1], lane >> 2,
           blk[2 + (lane >> 2)]);
}

static float rnd_scale(void) {
    /* random fp16 scale, including subnormals and negatives */
    uint16_t h = (uint16_t)(rnd() & 0xFFFF);
    switch (rnd() % 4) {
        case 0: h &= 0x83FF; break;          /* small / subnormal */
        case 1: h = (uint16_t)((h & 0x8000) | 0x3C00 | (h & 0x03FF)); break;
        default: break;
    }
    return bonsai_fp16_to_fp32(h);
}

/* Decoder sweep: compare pq2_0_dequant_block_imci against pq2_0_dequant_block
 * for adversarial byte patterns and random blocks, at all 64 source offsets. */
static void decoder_sweep(void) {
    printf("=== decoder: IMCI vs scalar, adversarial + random ===\n");
    uint8_t arena[34 * 4 + 256];
    long mism = 0, total = 0, nancases = 0;
    float a[QK_PQ2_0];
    static float b[QK_PQ2_0] __attribute__((aligned(64)));
    static const int pats[] = { 0x00, 0xFF, 0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC };
    for (unsigned pi = 0; pi < sizeof(pats) / sizeof(pats[0]); ++pi) {
        for (int off = 0; off < 64; ++off) {
            uint8_t *blk = arena + off;
            for (int i = 0; i < 34; ++i) blk[i] = (uint8_t)pats[pi];
            blk[0] = 0x00; blk[1] = 0x3C;            /* d = 1.0 */
            pq2_0_dequant_block(blk, a);
            pq2_0_dequant_block_imci(blk, b);
            for (int i = 0; i < QK_PQ2_0; ++i) {
                ++total;
                if (!same_bits(a[i], b[i])) { ++mism; report_mismatch("pat", blk, i, a[i], b[i]); }
            }
        }
    }
    for (int t = 0; t < 5000; ++t) {
        uint8_t *blk = arena + (rnd() % 64);
        for (int i = 0; i < 34; ++i) blk[i] = (uint8_t)rnd();
        pq2_0_dequant_block(blk, a);
        pq2_0_dequant_block_imci(blk, b);
        int isnan_d = (a[0] != a[0]);
        if (isnan_d) ++nancases;
        for (int i = 0; i < QK_PQ2_0; ++i) {
            ++total;
            if (!same_bits(a[i], b[i])) { ++mism; report_mismatch("rnd", blk, i, a[i], b[i]); }
        }
    }
    printf("  %ld comparisons, %ld mismatches (%ld random blocks had a NaN scale)\n",
           total, mism, nancases);
}

/* Build a synthetic PQ2_0 matrix (d rows of n weights) in a guard-slack buffer. */
static uint8_t *mkmatrix(int n, int d, size_t *sch, size_t *slack) {
    size_t rowb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    size_t need = rowb * (size_t)d;
    *slack = 512;
    *sch = need + *slack;
    uint8_t *w = (uint8_t *)malloc(*sch);
    for (size_t i = 0; i < need; ++i) w[i] = (uint8_t)rnd();
    /* give every block a finite scale */
    for (int i = 0; i < d; ++i) {
        uint8_t *row = w + (size_t)i * rowb;
        for (int blk = 0; blk * QK_PQ2_0 < n; ++blk) {
            uint16_t h = (uint16_t)(0x3800 | (rnd() & 0x03FF));
            row[blk * PQ2_0_BLOCK_BYTES + 0] = (uint8_t)(h & 0xFF);
            row[blk * PQ2_0_BLOCK_BYTES + 1] = (uint8_t)(h >> 8);
        }
    }
    for (size_t i = need; i < *sch; ++i) w[i] = 0xA5;     /* poison slack */
    return w;
}

static void matvec_shapes(void) {
    printf("\n=== matvec: knc vs ref across shapes ===\n");
    const int ns[] = { 128, 256, 384, 512, 1024, 5120, 6144, 17408, 130, 136,
                       200, 192, 129, 255, 5, 127 };
    const int ds[] = { 1, 3, 24 };
    for (unsigned a = 0; a < sizeof(ns)/sizeof(ns[0]); ++a) {
        for (unsigned b = 0; b < sizeof(ds)/sizeof(ds[0]); ++b) {
            const int n = ns[a], d = ds[b];
            size_t sch, slack;
            uint8_t *w = mkmatrix(n, d, &sch, &slack);
            float *x  = xbuf;
            float *y1 = (float *)malloc(sizeof(float) * (size_t)d);
            float *y2 = (float *)malloc(sizeof(float) * (size_t)d);
            for (int i = 0; i < n + 16; ++i) x[i] = (float)((int)(rnd() % 2000) - 1000) * 0.001f;
            pq2_0_matvec_ref(y1, x, w, n, d);
            pq2_0_matvec_knc(y2, x, w, n, d);
            float amax = 0.0f, rmax = 0.0f;
            for (int i = 0; i < d; ++i) {
                float e = fabsf(y1[i] - y2[i]);
                if (e > amax) amax = e;
                float den = fabsf(y1[i]) > 1e-30f ? fabsf(y1[i]) : 1e-30f;
                if (e / den > rmax) rmax = e / den;
            }
            printf("  n=%-6d d=%-3d  max_abs=%.3e  max_rel=%.3e  %s\n",
                   n, d, amax, rmax, rmax < 1e-3f ? "ok" : "DIFF");
            free(w); free(y1); free(y2);
        }
    }
}

/* Does the poison in the 512-byte slack after the last block change anything? */
static void slack_invariance(void) {
    printf("\n=== does poison in the trailing slack change the matvec result? ===\n");
    const int n = 512, d = 4;
    size_t sch, slack;
    uint8_t *w = mkmatrix(n, d, &sch, &slack);
    float *x = xbuf;
    float y1[4], y2[4];
    for (int i = 0; i < n; ++i) x[i] = 0.01f * (float)(int)(rnd() % 1000);
    pq2_0_matvec_knc(y1, x, w, n, d);
    for (size_t i = sch - slack; i < sch; ++i) w[i] = 0x5A;
    pq2_0_matvec_knc(y2, x, w, n, d);
    int diff = 0;
    for (int i = 0; i < d; ++i) if (y1[i] != y2[i]) ++diff;
    printf("  slack 0xA5 vs 0x5A: %d/%d rows differ\n", diff, d);
    free(w);
}

int main(void) {
    decoder_sweep();
    matvec_shapes();
    slack_invariance();
    return 0;
}
