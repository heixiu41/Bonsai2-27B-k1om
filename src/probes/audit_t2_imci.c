/* audit_t2_imci.c -- host execution of the REAL KNC decoder/kernel in pq2_0.h,
 * compiled through the instrumentation shim (D:\zigtest\shim\immintrin.h).
 *
 *   zig cc -O1 -std=c11 -o D:\zigtest\t2.exe audit_t2_imci.c -I . -I D:\zigtest\shim -lm
 *
 * Answers, by measurement:
 *   (a) is pq2_0_dequant_block_imci bit-identical to pq2_0_dequant_block, for
 *       every one of the 64 possible byte phases of a block?
 *   (b) how far past the end of a 34-byte block does a decode read, worst phase?
 *   (c) do all eight _mm512_load_ps(x + j + k) offsets satisfy the 64-byte
 *       alignment rule (k = 0,16,32,48,64,80,96,112), and what happens if x is
 *       only 16-byte aligned (the mm_plain()/mm_folded(signs=NULL) hazard)?
 *   (d) does pq2_0_matvec_knc consume every element of every row exactly once?
 *       (all-codes-0 weights + x == 1 must give exactly -n; codes==2 gives +n)
 */
#define __MIC__ 1
#include "pq2_0.h"

#include <stdio.h>
#include <string.h>

audit_touch_t audit_touch;
audit_align_t audit_align;
volatile int  audit_sink;

static uint32_t rng = 0x12345678u;
static uint32_t xs(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

/* 64-byte aligned arena so the byte phase of a block can be set exactly */
static uint8_t arena[8192] __attribute__((aligned(64)));

/* finite fp16 (exp 0..30 -> subnormals, normals) + 32 random code bytes */
static void fill_block(uint8_t *b)
{
    const uint16_t h = (uint16_t)(((xs() % 31u) << 10) | (xs() & 0x3ffu) |
                                  ((xs() & 1u) << 15));
    b[0] = (uint8_t)(h & 0xff);
    b[1] = (uint8_t)(h >> 8);
    for (int i = 2; i < 34; ++i) b[i] = (uint8_t)xs();
    /* fill the rest of the enclosing 64-byte blocks with junk: the decoder
     * must not depend on it (it must only deliver bytes [blk,blk+34)) */
}

static long decode_phase_failures = 0;
static long decode_cases = 0;

static void test_decode_exact(void)
{
    float a[QK_PQ2_0] __attribute__((aligned(64)));
    float b[QK_PQ2_0] __attribute__((aligned(64)));
    for (int phase = 0; phase < 64; ++phase) {
        uint8_t *blk = arena + 64 + phase;         /* aligned arena + phase */
        for (int rep = 0; rep < 256; ++rep) {
            fill_block(blk);
            pq2_0_dequant_block(blk, a);
            pq2_0_dequant_block_imci(blk, b);
            decode_cases++;
            for (int j = 0; j < QK_PQ2_0; ++j) {
                uint32_t ua, ub;
                memcpy(&ua, &a[j], 4); memcpy(&ub, &b[j], 4);
                if (ua != ub) {
                    if (decode_phase_failures < 5)
                        printf("   decoder mismatch phase=%d rep=%d j=%d scalar=%a imci=%a\n",
                               phase, rep, j, (double)a[j], (double)b[j]);
                    decode_phase_failures++;
                    break;
                }
            }
        }
    }
    printf("(a) pq2_0_dequant_block_imci vs scalar: %ld block decodes over all 64 phases,"
           " %ld mismatching blocks\n", decode_cases, decode_phase_failures);
}

static void test_overread(void)
{
    uintptr_t worst_after = 0, worst_before = 0;
    int worst_phase_a = -1, worst_phase_b = -1;
    long max_block_bytes = 0;
    float b[QK_PQ2_0] __attribute__((aligned(64)));

    for (int phase = 0; phase < 64; ++phase) {
        uint8_t *blk = arena + 256 + phase;
        memset(arena, 0x5a, sizeof arena);
        fill_block(blk);
        audit_touch.active = 1;
        audit_touch.lo = (uintptr_t)-1; audit_touch.hi = 0; audit_touch.bytes = 0;
        pq2_0_dequant_block_imci(blk, b);
        audit_touch.active = 0;

        const uintptr_t begin = (uintptr_t)blk;
        const uintptr_t end   = begin + PQ2_0_BLOCK_BYTES;      /* 34 */
        const uintptr_t after  = (audit_touch.hi + 1 > end) ? audit_touch.hi + 1 - end : 0;
        const uintptr_t before = (audit_touch.lo < begin) ? begin - audit_touch.lo : 0;
        if (after  > worst_after)  { worst_after  = after;  worst_phase_a = phase; }
        if (before > worst_before) { worst_before = before; worst_phase_b = phase; }
        if (audit_touch.hi + 1 - audit_touch.lo > (uintptr_t)max_block_bytes)
            max_block_bytes = (long)(audit_touch.hi + 1 - audit_touch.lo);
    }
    printf("(b) one decode of a 34-byte block reads a %ld-byte span; worst case reads\n"
           "    %llu bytes PAST the block end (blk byte-phase %d) and\n"
           "    %llu bytes BEFORE the block start (phase %d)\n",
           max_block_bytes,
           (unsigned long long)worst_after, worst_phase_a,
           (unsigned long long)worst_before, worst_phase_b);
}

/* --- the real kernel, called the way model.c calls it ----------------- */
static void test_kernel_align_and_math(void)
{
    const int n = 512, d = 4;                 /* 4 blocks per row */
    const size_t row_bytes = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)malloc((size_t)d * row_bytes + 512);   /* slack as the caller must */
    float *x64 = (float *)malloc(sizeof(float) * (n + 32) + 64);
    float *y = (float *)malloc(sizeof(float) * d);
    float *yr = (float *)malloc(sizeof(float) * d);
    memset(w, 0, (size_t)d * row_bytes + 512);          /* all codes 0 -> value -1 when d=1 */
    for (size_t i = 0; i < (size_t)d * row_bytes + 512; i += PQ2_0_BLOCK_BYTES) {
        w[i] = 0x00; w[i + 1] = 0x3C;                   /* fp16 1.0 */
    }
    float *xa = (float *)(((uintptr_t)x64 + 63) & ~(uintptr_t)63);
    /* 16-byte aligned but deliberately not 64: exactly what calloc/posix_memalign(16)
     * gives the model's other scratch buffers, i.e. what mm_plain()/mm_folded(x, NULL)
     * would hand to this kernel. */
    float *x16 = xa + 4;
    for (int i = 0; i < n; ++i) xa[i] = 1.0f;
    memcpy(x16, xa, sizeof(float) * n);

    /* (c1) 64-byte aligned x, exactly like mm_folded's fold_x */
    memset(&audit_align, 0, sizeof audit_align);
    pq2_0_matvec_knc(y, xa, w, n, d);
    printf("(c1) x 64-byte aligned : %ld vector accesses, %ld misaligned\n",
           audit_align.loads, audit_align.faults);
    printf("     y[0]=%a (expect %a for all-codes-0 weights = -n)\n",
           (double)y[0], (double)(-1.0f * n));

    /* (c2) x only 16-byte aligned: what mm_plain()/mm_folded(signs==NULL) would do */
    memset(&audit_align, 0, sizeof audit_align);
    pq2_0_matvec_knc(y, x16, w, n, d);
    printf("(c2) x 16-byte aligned : %ld vector accesses, %ld misaligned"
           " (first at +%ld mod 64) -> the card would fault\n",
           audit_align.loads, audit_align.faults, audit_align.first_bad_mod64);

    /* (d) exact coverage: every element consumed exactly once */
    long bad_neg = 0, bad_pos = 0, cases = 0;
    const int ns[] = { 0,1,2,127,128,129,200,255,256,257,300,384,512,1000,1024,5120 };
    for (unsigned k = 0; k < sizeof ns / sizeof ns[0]; ++k) {
        const int nn = ns[k];
        const size_t rb = (size_t)((nn + QK_PQ2_0 - 1) / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        uint8_t *ww = (uint8_t *)malloc(rb * 2 + 512);
        float *xx = (float *)malloc(sizeof(float) * (nn + 16) + 64);
        float *xxa = (float *)(((uintptr_t)xx + 63) & ~(uintptr_t)63);
        float yy[2], yy2[2];
        for (int sign = 0; sign < 2; ++sign) {
            /* 0x00 -> every 2-bit field is 0 -> value -1; 0xAA = 10 10 10 10 -> +1 */
            memset(ww, sign ? 0xAA : 0x00, rb * 2 + 512);
            for (size_t i = 0; i + 1 < rb; i += PQ2_0_BLOCK_BYTES) {
                ww[i] = 0x00; ww[i + 1] = 0x3C;            /* d = 1.0 */
            }
            for (int i = 0; i < nn; ++i) xxa[i] = 1.0f;
            pq2_0_matvec_knc(yy, xxa, ww, nn, 1);
            pq2_0_matvec_ref(yy2, xxa, ww, nn, 1);
            const float expect = (sign ? 1.0f : -1.0f) * (float)nn;
            cases++;
            if (yy[0] != expect) { bad_neg += (sign == 0); bad_pos += (sign == 1);
                if (bad_neg + bad_pos < 6)
                    printf("   coverage n=%d sign=%d got %a want %a (ref %a)\n",
                           nn, sign, (double)yy[0], (double)expect, (double)yy2[0]); }
            if (yy2[0] != expect) printf("   REF disagrees n=%d got %a want %a\n",
                                         nn, (double)yy2[0], (double)expect);
        }
        free(ww); free(xx);
    }
    printf("(d) exact element coverage: %ld kernel runs over n=0..5120,"
           " %ld wrong results (double-count/skip would show here)\n",
           cases, bad_neg + bad_pos);

    /* (e) random comparison against the scalar kernel. float summation order
     * differs by construction, so the error is measured against the sum of the
     * term magnitudes (backward-error style), not against |y|. */
    long mismatch = 0, runs = 0;
    double worst_ratio = 0.0;
    for (int rep = 0; rep < 40; ++rep) {
        const int nn = 128 * (1 + (int)(xs() % 12));
        const int dd = 1 + (int)(xs() % 4);
        const size_t rb = (size_t)(nn / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        uint8_t *ww = (uint8_t *)malloc((size_t)dd * rb + 512);
        float *xx = (float *)malloc(sizeof(float) * nn + 64);
        float *xxa = (float *)(((uintptr_t)xx + 63) & ~(uintptr_t)63);
        float *yv = (float *)malloc(sizeof(float) * dd);
        float *yrf = (float *)malloc(sizeof(float) * dd);
        for (int b = 0; b < dd; ++b)
            for (long i = 0; i < (long)rb; i += PQ2_0_BLOCK_BYTES)
                fill_block(ww + (size_t)b * rb + i);
        for (int i = 0; i < nn; ++i) xxa[i] = (float)((int)(xs() % 2001) - 1000) / 512.0f;
        pq2_0_matvec_knc(yv, xxa, ww, nn, dd);
        pq2_0_matvec_ref(yrf, xxa, ww, nn, dd);
        for (int b = 0; b < dd; ++b) {
            double tsum = 0.0;
            float wfb[QK_PQ2_0] __attribute__((aligned(64)));
            const uint8_t *row = ww + (size_t)b * rb;
            for (int j = 0; j < nn; j += QK_PQ2_0) {
                pq2_0_dequant_block(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wfb);
                for (int k = 0; k < QK_PQ2_0; ++k)
                    tsum += fabs((double)wfb[k] * (double)xxa[j + k]);
            }
            if (tsum <= 0.0) continue;
            const double err = fabs((double)yv[b] - (double)yrf[b]) / tsum;
            if (err > worst_ratio) worst_ratio = err;
            if (err > 1e-5) mismatch++;
        }
        runs++;
        free(ww); free(xx); free(yv); free(yrf);
    }
    printf("(e) random knc-vs-ref: %ld runs, %ld rows above 1e-5 of sum|w*x|"
           " (worst %.3g)\n", runs, mismatch, worst_ratio);

    free(w); free(x64); free(y); free(yr);
}

int main(void)
{
    printf("T2 IMCI decoder/kernel audit (host emulation of the documented KNC semantics)\n");
    test_decode_exact();
    test_overread();
    test_kernel_align_and_math();
    return 0;
}
