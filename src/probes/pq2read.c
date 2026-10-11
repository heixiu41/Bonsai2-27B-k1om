/*
 * pq2read.c  -- review probe (k1om): measure the exact read footprint of
 * pq2_0_dequant_block_imci() and prove whether bytes outside the 34-byte
 * PQ2_0 block can change the decoded values.
 *
 * Method: 4 anonymous pages, the third mprotect(PROT_NONE) as a guard. A fake
 * 34-byte block is placed so that its last byte sits `dist` bytes before the
 * guard. The decoder runs in a forked child, so a fault is observable per
 * distance instead of killing the sweep.
 *
 * Not linked against gguf.c/model.c: needs only src/bonsai/pq2_0.h.
 *   icc -mmic -O2 -std=c11 pq2read.c -o pq2read -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pq2_0.h"

#define PAGE 4096

static void run_child(const uint8_t *blk, float *out) {
    pid_t pid = fork();
    if (pid == 0) {
        pq2_0_dequant_block_imci(blk, out);
        volatile float s = 0.0f;
        for (int i = 0; i < QK_PQ2_0; ++i) s += out[i];
        (void)s;
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) printf("SIGNAL(%d)", WTERMSIG(st));
    else                 printf("ok");
}

int main(void) {
    uint8_t *base = (uint8_t *)mmap(NULL, 4 * PAGE, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    uint8_t *guard = base + 2 * PAGE;
    if (mprotect(guard, PAGE, PROT_NONE) != 0) { perror("mprotect"); return 1; }
    float *out = (float *)(base + PAGE);        /* page-aligned -> 64-aligned */

    printf("=== A. minimum readable slack needed after a 34-byte PQ2_0 block ===\n");
    printf("(block occupies [guard-34-dist, guard-dist); fault => decoder read\n"
           " at or past the guard page)\n");
    int first_ok = -1;
    for (int dist = 0; dist <= 160; ++dist) {
        uint8_t *blk = guard - 34 - dist;
        for (int i = 0; i < 34; ++i) blk[i] = (uint8_t)(0x3C00u >> 8);
        blk[0] = 0x00; blk[1] = 0x3C;            /* d = 1.0h */
        for (int i = 2; i < 34; ++i) blk[i] = (uint8_t)(i * 37 + 11);
        printf("  dist=%3d  last_block_byte=guard-%3d  qs&63=%2d  -> ",
               dist, dist, (int)(((uintptr_t)(blk + 2)) & 63));
        run_child(blk, out);
        printf("\n");
        fflush(stdout);
    }
    /* report the first distance that survives, using a coarse scan */
    for (int dist = 0; dist <= 200; ++dist) {
        uint8_t *blk = guard - 34 - dist;
        for (int i = 0; i < 34; ++i) blk[i] = (uint8_t)(i * 37 + 11);
        blk[0] = 0x00; blk[1] = 0x3C;
        pid_t pid = fork();
        if (pid == 0) { pq2_0_dequant_block_imci(blk, out); _exit(0); }
        int st = 0; waitpid(pid, &st, 0);
        if (!WIFSIGNALED(st)) { first_ok = dist; break; }
    }
    printf("\n  => first fault-free trailing distance: %d bytes\n", first_ok);
    printf("  => decoder reads up to %d bytes from the start of a block (%d past its 34-byte end)\n",
           first_ok + 34, first_ok);

    printf("\n=== B. do the bytes past the 16-byte payload change the result? ===\n");
    {
        int dist = 300;                           /* comfortably readable slack */
        uint8_t *blk = guard - 34 - dist;
        for (int i = 0; i < 34; ++i) blk[i] = (uint8_t)(i * 37 + 11);
        blk[0] = 0x00; blk[1] = 0x3C;
        float ref[QK_PQ2_0], got[QK_PQ2_0];
        pq2_0_dequant_block(blk, ref);
        /* NOTE: no fork here -- the child's writes would land in its own COW
         * copy and the parent would compare uninitialised memory. */
        memset(blk + 34, 0x00, dist);
        pq2_0_dequant_block_imci(blk, got);
        int m0 = 0;
        for (int i = 0; i < QK_PQ2_0; ++i) if (ref[i] != got[i]) ++m0;
        memset(blk + 34, 0xA5, dist);
        pq2_0_dequant_block_imci(blk, got);
        int m1 = 0;
        for (int i = 0; i < QK_PQ2_0; ++i) if (ref[i] != got[i]) ++m1;
        printf("  slack 0x00: %d/128 differ from scalar\n", m0);
        printf("  slack 0xA5: %d/128 differ from scalar\n", m1);

        /* worst case for correctness: only the 48 bytes the decoder actually
         * touches are readable at all, and they are poison. */
        printf("  (poison is confined to bytes %d..%d past the block start)\n",
               34, dist + 34);
    }

    printf("\n=== C. is _mm512_store_ps(out) really 64-byte-aligned only? ===\n");
    {
        /* misalign the destination by 4 bytes and see whether the store faults */
        float *bad = (float *)(base + PAGE + 4);
        uint8_t *blk = guard - 34 - 300;
        for (int i = 0; i < 34; ++i) blk[i] = (uint8_t)(i * 37 + 11);
        pid_t pid = fork();
        if (pid == 0) { pq2_0_dequant_block_imci(blk, bad); _exit(0); }
        int st = 0; waitpid(pid, &st, 0);
        printf("  out aligned to 4 bytes : %s\n", WIFSIGNALED(st) ? "SIGNAL" : "ok");
        pid = fork();
        if (pid == 0) { pq2_0_dequant_block_imci(blk, (float *)(base + PAGE)); _exit(0); }
        waitpid(pid, &st, 0);
        printf("  out aligned to 4096    : %s\n", WIFSIGNALED(st) ? "SIGNAL" : "ok");
    }

    printf("\n=== E. does pq2_0_matvec_knc require a 64-byte aligned x? ===\n");
    {
        /* one full row: 128 weights = 1 block, w has plenty of slack */
        uint8_t *w = (uint8_t *)(base + PAGE);         /* page 1, unused by B/C */
        for (int i = 0; i < 64; ++i) w[i] = (uint8_t)(i * 13 + 7);
        w[0] = 0x00; w[1] = 0x3C;
        float *xmis = (float *)(base + PAGE + 16);     /* 64+16 -> 16-byte aligned */
        float *xok  = (float *)(base + PAGE + 2048);   /* 64-byte aligned, in page 1 */
        float y;
        for (int i = 0; i < 128; ++i) xok[i] = 0.5f;
        pid_t pid = fork();
        if (pid == 0) { pq2_0_matvec_knc(&y, xok, w, QK_PQ2_0, 1); _exit(0); }
        int st = 0; waitpid(pid, &st, 0);
        printf("  x 64-byte aligned (fold_x-like) : %s\n", WIFSIGNALED(st) ? "SIGNAL" : "ok");
        pid = fork();
        if (pid == 0) { pq2_0_matvec_knc(&y, xmis, w, QK_PQ2_0, 1); _exit(0); }
        waitpid(pid, &st, 0);
        printf("  x 16-byte aligned (calloc/malloc): %s\n", WIFSIGNALED(st) ? "SIGNAL" : "ok");
    }
    return 0;
}
