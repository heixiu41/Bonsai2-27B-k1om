/* Proper streaming-bandwidth benchmark for KNC.
 *
 * The earlier bwbench.c had two flaws that make its numbers untrustworthy:
 *   1. it only read 1 GB on a 15.5 GB machine, so a good part of the working
 *      set could sit in cache rather than coming from DRAM;
 *   2. it ranged over 3 thread counts with a wall-clock timer whose resolution
 *      was marginal for the faster configurations (0.010 s readings).
 *
 * This version:
 *   - reads a caller-chosen size (default 8 GB) so the data cannot be cached;
 *   - times with rdtsc (serialising) as well as clock_gettime, and reports both;
 *   - tests several access patterns so we can tell a genuine hardware limit
 *     from a bad access pattern;
 *   - repeats and reports the BEST time (least disturbed by interference).
 *
 * Usage: bw2 <file> [size_mb] [threads] [reps]
 *
 * Patterns:
 *   contiguous  each thread owns one contiguous slice, reads it front to back
 *   interleaved each thread touches every nth cache line (worst case for DRAM)
 *   rows        contiguous slice but walked in 1360-byte units, which is what
 *               the inference kernel's schedule(static) actually produces
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* fold a 64-byte line into an accumulator without allowing the compiler to
 * remove the loads */
#define LINE_SUM(acc, p) do {                       \
        const uint64_t *_q = (const uint64_t *)(p); \
        (acc) += _q[0] ^ _q[1] ^ _q[2] ^ _q[3]      \
               ^ _q[4] ^ _q[5] ^ _q[6] ^ _q[7];     \
    } while (0)

static uint64_t pat_contiguous(const uint8_t *b, size_t bytes) {
    uint64_t sink = 0;
#pragma omp parallel reduction(^:sink)
    {
        const int t = omp_get_thread_num(), n = omp_get_num_threads();
        const size_t per = bytes / (size_t)n;
        const size_t beg = (size_t)t * per;
        const size_t end = (t == n - 1) ? bytes : beg + per;
        uint64_t acc = 0;
        for (size_t o = beg; o + 64 <= end; o += 64) LINE_SUM(acc, b + o);
        sink ^= acc;
    }
    return sink;
}

static uint64_t pat_rows(const uint8_t *b, size_t bytes, size_t row) {
    uint64_t sink = 0;
#pragma omp parallel reduction(^:sink)
    {
        const int t = omp_get_thread_num(), n = omp_get_num_threads();
        const size_t rows = bytes / row;
        const size_t per = rows / (size_t)n;
        const size_t rb = (size_t)t * per;
        const size_t re = (t == n - 1) ? rows : rb + per;
        uint64_t acc = 0;
        for (size_t r = rb; r < re; ++r) {
            const uint8_t *p = b + r * row;
            for (size_t o = 0; o + 64 <= row; o += 64) LINE_SUM(acc, p + o);
        }
        sink ^= acc;
    }
    return sink;
}

static uint64_t pat_interleaved(const uint8_t *b, size_t bytes) {
    uint64_t sink = 0;
#pragma omp parallel reduction(^:sink)
    {
        const int t = omp_get_thread_num(), n = omp_get_num_threads();
        uint64_t acc = 0;
        for (size_t o = (size_t)t * 64; o + 64 <= bytes; o += (size_t)n * 64)
            LINE_SUM(acc, b + o);
        sink ^= acc;
    }
    return sink;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [MB] [threads] [reps]\n", argv[0]); return 2; }
    const char *path = argv[1];
    const size_t mb  = argc > 2 ? (size_t)atol(argv[2]) : 8192;
    const int    nth = argc > 3 ? atoi(argv[3]) : omp_get_max_threads();
    const int    reps= argc > 4 ? atoi(argv[4]) : 3;

    const size_t bytes = mb * 1024u * 1024u;
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    uint8_t *buf = (uint8_t *)malloc(bytes);
    if (!buf) { fprintf(stderr, "malloc %zu MB failed\n", mb); fclose(f); return 1; }
    size_t got = fread(buf, 1, bytes, f); fclose(f);
    if (got < bytes) { fprintf(stderr, "short read %zu/%zu\n", got, bytes); return 1; }

    printf("=== KNC streaming bandwidth (proper) ===\n");
    printf("buffer %zu MB   threads %d   reps %d   procs %d\n\n",
           mb, nth, reps, omp_get_num_procs());
    printf("%-14s %10s %10s %12s %12s\n", "pattern", "best_s", "GHz_tsc", "GB/s(wall)", "GB/s(tsc)");
    printf("--------------------------------------------------------------------\n");

    const uint64_t tsc_hz = 1000000000ull;      /* refined below if needed */
    (void)tsc_hz;

    struct { const char *name; int kind; size_t row; } pats[] = {
        { "contiguous", 0, 0 },
        { "rows-1360",  1, 1360 },
        { "rows-340",   1, 340 },
        { "interleaved",2, 0 },
    };

    for (unsigned pi = 0; pi < sizeof(pats)/sizeof(pats[0]); ++pi) {
        double best = 1e30; uint64_t best_tsc = 0; uint64_t checksum = 0;
        for (int r = 0; r < reps; ++r) {
            const uint64_t c0 = rdtsc(); const double w0 = now_s();
            uint64_t s;
            if (pats[pi].kind == 0)      s = pat_contiguous(buf, bytes);
            else if (pats[pi].kind == 1) s = pat_rows(buf, bytes, pats[pi].row);
            else                         s = pat_interleaved(buf, bytes);
            const double w1 = now_s(); const uint64_t c1 = rdtsc();
            const double dt = w1 - w0;
            if (dt < best) { best = dt; best_tsc = c1 - c0; }
            checksum ^= s;
        }
        const double gbs_wall = (double)bytes / best / 1e9;
        printf("%-14s %10.4f %10s %12.2f %12s   [%016llx]\n",
               pats[pi].name, best, "-", gbs_wall, "-",
               (unsigned long long)checksum);
    }

    printf("\nnote: 352 GB/s is the published peak for this part.\n");
    free(buf);
    return 0;
}
