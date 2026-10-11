/* audit_t3_guard.c -- tail paths, row stride, and guard-page proof of reads.
 *
 *   zig cc -O2 -std=c11 -o D:\zigtest\t3.exe audit_t3_guard.c -I . -lm
 *   D:\zigtest\t3.exe <mode>       (each mode is a separate process: some crash)
 *
 * Modes:
 *   ref_exact    n=256 (n%128==0), weights allocation ends exactly at a
 *                PAGE_NOACCESS guard page -> must NOT fault.
 *   ref_tail     n=200 (n%128!=0), buffer = d*floor(n/128)*34 bytes ending at
 *                the guard -> shows whether the scalar kernel's *tail* reads
 *                past the end of the buffer the kernel's own stride implies.
 *   knc_s48      last block at byte-phase 46 mod 64, 48 bytes of slack before
 *                the guard -> expect a fault (over-read).
 *   knc_s112     same byte-phase, 112 bytes of slack -> expect survival.
 *   tail_dep     n=200, d=2, padded: shows whether row 0's tail value depends
 *                on row 1's first block (it must not).
 *   ceil_layout  n=200, d=2 laid out with ceil(n/128)*34 = 68 bytes/row (the
 *                only self-consistent layout for a non-multiple-of-128 width);
 *                compares pq2_0_matvec_ref with an independent reference.
 */
#define __MIC__ 1
#include "pq2_0.h"

#include <stdio.h>
#include <string.h>

/* Minimal Win32 declarations: including <windows.h> here would pull in the real
 * <immintrin.h> and clash with the audit shim, so declare just what we need. */
__declspec(dllimport) void *__stdcall VirtualAlloc(void *addr, size_t size,
                                                   unsigned long type, unsigned long protect);
__declspec(dllimport) int __stdcall VirtualProtect(void *addr, size_t size,
                                                   unsigned long newp, unsigned long *oldp);
#define AUDIT_MEM_COMMIT    0x1000ul
#define AUDIT_MEM_RESERVE   0x2000ul
#define AUDIT_PAGE_READWRITE 0x04ul
#define AUDIT_PAGE_NOACCESS  0x01ul
#define AUDIT_PAGE_SIZE      4096ul   /* x86-64 Windows page size */

audit_touch_t audit_touch;
audit_align_t audit_align;
volatile int  audit_sink;

static uint8_t *guard_region(size_t usable, size_t *usable_out)
{
    const size_t ps = AUDIT_PAGE_SIZE;
    size_t npages = (usable + ps - 1) / ps + 1;
    uint8_t *base = (uint8_t *)VirtualAlloc(NULL, npages * ps,
                                            AUDIT_MEM_RESERVE | AUDIT_MEM_COMMIT,
                                            AUDIT_PAGE_READWRITE);
    if (!base) { printf("VirtualAlloc failed\n"); exit(2); }
    unsigned long old = 0;
    VirtualProtect(base + (npages - 1) * ps, ps, AUDIT_PAGE_NOACCESS, &old);
    *usable_out = (npages - 1) * ps;
    return base;
}

static uint32_t rng = 0xC0FFEEu;
static uint32_t xs(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void fill_codes(uint8_t *b, size_t nb)
{
    for (size_t i = 0; i + 1 < nb; i += PQ2_0_BLOCK_BYTES) {
        b[i] = 0x00; b[i + 1] = 0x3C;                 /* d = 1.0 */
        for (size_t k = 2; k < PQ2_0_BLOCK_BYTES && i + k < nb; ++k) b[i + k] = (uint8_t)xs();
    }
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "ref_exact";
    float y[8];
    float x[2048];
    volatile float sink = 0.0f;

    if (!strcmp(mode, "ref_exact")) {
        const int n = 256, d = 4;
        const size_t need = (size_t)d * (n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;  /* 4*2*34=272 */
        size_t usable = 0;
        uint8_t *base = guard_region(need, &usable);
        uint8_t *w = base + usable - need;            /* end exactly at the guard */
        fill_codes(w, need);
        for (int i = 0; i < n; ++i) x[i] = 0.5f;
        pq2_0_matvec_ref(y, x, w, n, d);
        sink = y[0] + y[1] + y[2] + y[3];
        printf("ref_exact: survived, y=%.6f %.6f %.6f %.6f (guard page untouched)\n",
               (double)y[0], (double)y[1], (double)y[2], (double)y[3]);
        return 0;
    }

    if (!strcmp(mode, "ref_tail")) {
        const int n = 200, d = 2;                     /* floor(200/128)=1 block/row */
        const size_t need = (size_t)d * (n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;  /* 68 */
        size_t usable = 0;
        uint8_t *base = guard_region(need, &usable);
        uint8_t *w = base + usable - need;
        fill_codes(w, need);
        for (int i = 0; i < n; ++i) x[i] = 0.5f;
        printf("ref_tail: about to call pq2_0_matvec_ref(n=200,d=2) on a %zu-byte buffer\n", need);
        fflush(stdout);
        pq2_0_matvec_ref(y, x, w, n, d);
        sink = y[0] + y[1];
        printf("ref_tail: survived (no read past the buffer end)\n");
        return 0;
    }

    if (!strcmp(mode, "knc_s48") || !strcmp(mode, "knc_s112")) {
        const int slack = (mode[5] == '4') ? 48 : 112;
        const size_t block = PQ2_0_BLOCK_BYTES;                  /* one 34-byte block */
        const size_t need = block + slack;                       /* end at guard - slack */
        size_t usable = 0;
        uint8_t *base = guard_region(need, &usable);
        uint8_t *w = base + usable - need;                       /* block at phase 46 */
        fill_codes(w, block);
        float out[QK_PQ2_0] __attribute__((aligned(64)));
        printf("knc_%s: block at phase %llu, %d bytes of slack to the guard\n",
               mode + 4, (unsigned long long)((uintptr_t)w & 63u), slack);
        fflush(stdout);
        pq2_0_dequant_block_imci(w, out);
        sink = out[0] + out[127];
        printf("knc_%s: survived the decode\n", mode + 4);
        return 0;
    }

    if (!strcmp(mode, "tail_dep")) {
        const int n = 200, d = 2;
        const size_t rb = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;   /* 34 */
        uint8_t *w = (uint8_t *)calloc(1, (size_t)d * rb + 512);
        float yy[2], zz[2];
        for (int i = 0; i < n; ++i) x[i] = 1.0f;
        fill_codes(w, (size_t)d * rb);
        pq2_0_matvec_ref(yy, x, w, n, d);
        /* now change ONLY row 1's first block (bytes [34,68)) */
        for (int k = 2; k < 34; ++k) w[34 + k] ^= 0xFF;
        pq2_0_matvec_ref(zz, x, w, n, d);
        printf("tail_dep: rows %d, weights stride %zu B/row, tail covers j=%d..%d\n",
               d, rb, (int)((n / QK_PQ2_0) * QK_PQ2_0), n - 1);
        printf("          y before %a %a\n          y after  %a %a\n",
               (double)yy[0], (double)yy[1], (double)zz[0], (double)zz[1]);
        printf("          row0 changed by mutating row1's first block: %s\n",
               (yy[0] != zz[0]) ? "YES -> row0's tail read row1's data" : "no");
        printf("          row1 changed: %s\n", (yy[1] != zz[1]) ? "YES" : "no");
        free(w);
        return 0;
    }

    if (!strcmp(mode, "ceil_layout")) {
        const int n = 200, d = 2;
        const int nblk = (n + QK_PQ2_0 - 1) / QK_PQ2_0;      /* 2 */
        const size_t stride = (size_t)nblk * PQ2_0_BLOCK_BYTES;   /* 68 */
        uint8_t *w = (uint8_t *)calloc(1, stride * d + 512);
        float want[2], got[2];
        for (int i = 0; i < n; ++i) x[i] = 1.0f;
        fill_codes(w, stride * d);
        /* independent reference assuming ceil(n/128) blocks per row */
        for (int r = 0; r < d; ++r) {
            float acc = 0.0f;
            float wb[QK_PQ2_0];
            for (int b = 0; b < nblk; ++b) {
                pq2_0_dequant_block(w + (size_t)r * stride + (size_t)b * PQ2_0_BLOCK_BYTES, wb);
                for (int k = 0; k < QK_PQ2_0; ++k) {
                    const int j = b * QK_PQ2_0 + k;
                    if (j < n) acc += wb[k] * x[j];
                }
            }
            want[r] = acc;
        }
        pq2_0_matvec_ref(got, x, w, n, d);
        printf("ceil_layout: stride %zu B/row, floor-stride used by the kernel = %zu B/row\n",
               stride, (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES);
        for (int r = 0; r < d; ++r)
            printf("   row %d: kernel %a   correct %a   %s\n", r,
                   (double)got[r], (double)want[r],
                   (got[r] == want[r]) ? "ok" : "WRONG");
        free(w);
        return 0;
    }

    printf("unknown mode %s\n", mode);
    return 2;
}
