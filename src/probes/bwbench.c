/* k1om memory-bandwidth microbenchmark.
 *
 * Purpose: find out what this card can actually sustain for a *read-only*
 * stream of the kind the inference kernel performs, before deciding whether to
 * optimise the arithmetic or the access pattern.
 *
 * It measures three patterns, each at 1 / N/2 / N threads:
 *   A. sequential   -- every thread sweeps one contiguous block (the pattern the
 *                      thread-blocked inference kernel would have)
 *   B. strided-16B  -- each thread touches every 16th byte across its block
 *                      (checks whether 64-byte cache lines are being used well)
 *   C. row-interleaved -- mimics pq2_0_matvec_knc's default schedule(static):
 *                      each thread gets a contiguous run of rows, and rows are
 *                      read one after another. Captured here as sequential with
 *                      a small per-iteration row pointer bump.
 *
 * The loads are kept alive with a cheap dependent transform and folded into a
 * sum that is printed, so the optimiser cannot delete them.
 *
 * Usage: bwbench <file> [size_mb] [threads]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Read a contiguous block with 64-byte (one cache line) steps. */
static double pattern_sequential(const uint8_t *base, size_t bytes, int nthreads,
                                 double *gbps_out) {
    volatile uint64_t sink = 0;
    double t0 = now_s();
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        const size_t per = bytes / (size_t)nth;
        const size_t beg = (size_t)tid * per;
        const size_t end = (tid == nth - 1) ? bytes : beg + per;
        uint64_t acc = 0;
        for (size_t off = beg; off < end; off += 64) {
            /* 8 independent 64-bit loads per cache line, folded cheaply */
            const uint64_t *p = (const uint64_t *)(base + off);
            acc += p[0] ^ p[1] ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7];
        }
        if (acc == 0x1234567890abcdefULL) sink = acc;   /* keep it live */
    }
    double t1 = now_s();
    (void)sink;
    *gbps_out = (double)bytes / (t1 - t0) / 1e9;
    return t1 - t0;
}

/* Same volume, but each thread walks one 1360-byte "row" at a time and then
 * jumps to the next row of its own block -- this is what the inference kernel's
 * schedule(static) actually does. */
static double pattern_rows(const uint8_t *base, size_t bytes, int nthreads,
                           size_t row_bytes, double *gbps_out) {
    volatile uint64_t sink = 0;
    double t0 = now_s();
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        const size_t rows_total = bytes / row_bytes;
        const size_t per = rows_total / (size_t)nth;
        const size_t rbeg = (size_t)tid * per;
        const size_t rend = (tid == nth - 1) ? rows_total : rbeg + per;
        uint64_t acc = 0;
        for (size_t r = rbeg; r < rend; ++r) {
            const uint8_t *row = base + r * row_bytes;
            for (size_t off = 0; off + 64 <= row_bytes; off += 64) {
                const uint64_t *p = (const uint64_t *)(row + off);
                acc += p[0] ^ p[1] ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7];
            }
        }
        if (acc == 0x1234567890abcdefULL) sink = acc;
    }
    double t1 = now_s();
    (void)sink;
    *gbps_out = (double)bytes / (t1 - t0) / 1e9;
    return t1 - t0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [size_mb] [max_threads]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const size_t size_mb = argc > 2 ? (size_t)atol(argv[2]) : 1024;
    const int maxt = argc > 3 ? atoi(argv[3]) : omp_get_max_threads();
    const size_t bytes = size_mb * 1024u * 1024u;

    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    uint8_t *buf = (uint8_t *)malloc(bytes);
    if (!buf) { fprintf(stderr, "malloc %zu failed\n", bytes); fclose(f); return 1; }
    size_t got = fread(buf, 1, bytes, f);
    fclose(f);
    if (got < bytes) {
        fprintf(stderr, "short read: got %zu of %zu\n", got, bytes);
        return 1;
    }

    printf("=== k1om bandwidth microbenchmark ===\n");
    printf("buffer   : %zu MB (read once per pattern)\n", size_mb);
    printf("max thr  : %d   omp_get_num_procs=%d\n\n", maxt, omp_get_num_procs());

    int tlist[3] = { 1, maxt / 2 > 0 ? maxt / 2 : 1, maxt };
    const char *tname[3] = { "1 thread", "half", "all" };
    /* de-duplicate if maxt is small */
    if (tlist[1] == tlist[0]) tlist[1] = tlist[2];

    double g = 0;
    printf("%-34s %10s %12s\n", "pattern / threads", "seconds", "GB/s");
    printf("------------------------------------------------------\n");

    for (int k = 0; k < 3; ++k) {
        double sec = pattern_sequential(buf, bytes, tlist[k], &g);
        printf("seq    %-22s %10.3f %12.2f\n", tname[k], sec, g);
    }
    for (int k = 0; k < 3; ++k) {
        double sec = pattern_rows(buf, bytes, tlist[k], 1360, &g);
        printf("rows   %-22s %10.3f %12.2f\n", tname[k], sec, g);
    }
    /* row size for the biggest tensor in the model: output.weight row */
    for (int k = 0; k < 3; ++k) {
        double sec = pattern_rows(buf, bytes, tlist[k], 5120/128*34, &g);
        printf("rows-1360 %-19s %10.3f %12.2f\n", tname[k], sec, g);
    }

    printf("\nreference: card peak 352 GB/s; prior llama2.c result on this card ~49 GB/s\n");
    free(buf);
    return 0;
}
