/* Measure OpenMP parallel-region overhead on k1om, with the real thread count
 * and the real work per call.
 *
 * THE QUESTION. The active PQ2_0 kernel, measured in isolation, scales to
 * 101.7 G weights/s at 122 threads (109x). The full model at 122 threads moves
 * only 8.78 G weights/s of weights -- 11.6x slower. Per token there are ~481
 * kernel calls, so the model spends 3.05 s / 481 = 6.3 ms per call while the
 * largest tensor's vector work is only 0.88 ms.
 *
 * Two candidate explanations, and they need different fixes:
 *   (I) PER-REGION OVERHEAD: entering/leaving a `#pragma omp parallel for` with
 *       122 threads costs milliseconds (thread wake-up, barrier, scheduler).
 *       Fix: one parallel region per layer, or persistent threads.
 *   (II) THE MEASUREMENT: the isolated benchmark is not comparable, because the
 *       model's calls interleave with other work and the working set differs.
 *
 * This measures (I) directly: time N parallel regions that do essentially no
 * work, and N regions that do a realistic amount (42 rows x 1360 bytes per
 * thread, which is what the largest model tensor gives each thread).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static volatile double sink;

/* A parallel region whose body is (nearly) empty: measures pure entry+exit. */
static double bench_empty(int nthreads, int nregions) {
    double acc = 0;
    const double t0 = now_s();
    for (int r = 0; r < nregions; ++r) {
        #pragma omp parallel num_threads(nthreads) reduction(+:acc)
        {
            acc += 1.0;
        }
    }
    const double t1 = now_s();
    sink += acc;
    return (t1 - t0) / nregions;          /* seconds per region */
}

/* A parallel for over rows, with each row doing a realistic amount of
 * arithmetic -- mimics the matvec's shape without needing the weights. */
static double bench_rows(int nthreads, int nregions, int rows, int rowfloats) {
    static float *buf = NULL;
    static int    bufn = 0;
    if (rows * rowfloats > bufn) {
        free(buf);
        buf = (float *)malloc((size_t)rows * rowfloats * sizeof(float));
        bufn = rows * rowfloats;
        for (int i = 0; i < bufn; ++i) buf[i] = (float)(i % 7) - 3.0f;
    }
    double acc = 0;
    const double t0 = now_s();
    for (int r = 0; r < nregions; ++r) {
        #pragma omp parallel for num_threads(nthreads) schedule(static) reduction(+:acc)
        for (int i = 0; i < rows; ++i) {
            float s = 0;
            const float *p = buf + (size_t)i * rowfloats;
            for (int j = 0; j < rowfloats; ++j) s += p[j];
            acc += s;
        }
    }
    const double t1 = now_s();
    sink += acc;
    return (t1 - t0) / nregions;
}

int main(int argc, char **argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 122;
    const int nr  = argc > 2 ? atoi(argv[2]) : 50;

    printf("=== OpenMP region overhead on k1om ===\n");
    printf("threads=%d  regions=%d  procs=%d\n\n", nth, nr, omp_get_num_procs());

    printf("%-38s %14s %14s\n", "case", "us/region", "total ms");

    /* 1. empty region -- pure entry/exit/barrier */
    {
        const double s = bench_empty(nth, nr);
        printf("%-38s %14.1f %14.1f\n",
               "empty region", s * 1e6, s * nr * 1e3);
    }

    /* 2. one row only: almost no work, reveals overhead floor with a for */
    {
        const double s = bench_rows(nth, nr, 1, 1360);
        printf("%-38s %14.1f %14.1f\n",
               "parallel for, 1 row", s * 1e6, s * nr * 1e3);
    }

    /* 3. 42 rows -- roughly what one thread gets in the largest tensor */
    {
        const double s = bench_rows(nth, nr, 42, 1360);
        printf("%-38s %14.1f %14.1f\n",
               "parallel for, 42 rows", s * 1e6, s * nr * 1e3);
    }

    /* 4. the same total work in ONE region rather than many */
    {
        const double t0 = now_s();
        double acc = 0;
        #pragma omp parallel num_threads(nth) reduction(+:acc)
        {
            #pragma omp for schedule(static)
            for (int r = 0; r < nr * 42; ++r) acc += 1.0;
        }
        const double t1 = now_s();
        sink += acc;
        printf("%-38s %14s %14.1f\n", "same work, ONE region", "-", (t1 - t0) * 1e3);
    }

    printf("\nsink %.3f\n", (double)sink);
    printf("\nIf 'empty region' is hundreds of us or more, per-region overhead\n");
    printf("dominates the model's 6.3 ms/call and the fix is fewer, larger regions.\n");
    return 0;
}
