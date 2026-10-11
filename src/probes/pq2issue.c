/* Measure KNC vector issue rate directly -- and find out whether a core's
 * second slot is reachable at all.
 *
 * WHY. The shipped decoder gets 0.918 G weights/s single-threaded; at ~40 vector
 * instructions per 128 weights that is 1.06 vector ops/cycle against a claimed
 * peak of 2/cycle. Two very different explanations:
 *   (U) we are not saturating the core -- independent work is available but the
 *       code does not expose it, so better scheduling/unrolling would help;
 *   (H) the core really cannot retire 2/cycle for this mix, and the ratio 1.0 is
 *       the hardware limit, so instruction COUNT is all that matters.
 * The sibling agents' microbenchmarks measured each op at ~2 cycles with 8
 * independent chains, which is consistent with either reading and does not
 * settle it. This does, by varying the number of INDEPENDENT chains (ILP) and
 * the number of THREADS per core (TLP) separately.
 *
 * Method, all on ONE core where possible (threads pinned by using few of them;
 * KNC schedules within a core first, so thread counts up to 4 tend to land on
 * one core -- verify with the printed core ids if /proc is readable):
 *   - for ILP: N independent accumulator chains, N = 1..16, timing the same
 *     total number of ADD operations. If the core can dual-issue, throughput
 *     rises with N until a plateau; the plateau value is the real issue width.
 *   - for TLP: repeat with 1, 2, 3, 4 threads.
 *   - do the same for the decoder's actual instruction mix rather than a pure
 *     chain of vpaddd, because vpermd/vpsrld may not dual-issue with each other.
 *
 * Every measured value is folded into a volatile sink so nothing is eliminated,
 * and the loop trip counts are printed so the arithmetic is checkable.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

static volatile float fsink;
static volatile int   isink;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define MHZ 1238.094
#define ITERS 4000000

/* N independent vpaddd chains. Total vpaddd executed = N * ITERS. */
static double bench_add(int nchain, int iters) {
    __m512 a[16];
    for (int i = 0; i < nchain; ++i) a[i] = _mm512_set1_ps((float)(i + 1));
    const __m512 one = _mm512_set1_ps(1.0f);
    const double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        /* the chain count is a compile-time-unrollable inner sequence */
        for (int i = 0; i < nchain; ++i) a[i] = _mm512_add_ps(a[i], one);
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int i = 0; i < nchain; ++i) s = _mm512_add_ps(s, a[i]);
    fsink = _mm512_reduce_add_ps(s);
    const double ops = (double)nchain * iters;
    return ops / (t1 - t0);          /* ops per second */
}

/* Rotate-style chains: vpermd, the instruction actually suspected of being slow */
static double bench_perm(int nchain, int iters) {
    __m512i a[16];
    for (int i = 0; i < nchain; ++i) a[i] = _mm512_set1_epi32(i + 1);
    const __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    const double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < nchain; ++i) a[i] = _mm512_permutevar_epi32(idx, a[i]);
    }
    const double t1 = now_s();
    __m512i s = _mm512_setzero_si512();
    for (int i = 0; i < nchain; ++i) s = _mm512_add_epi32(s, a[i]);
    isink = _mm512_reduce_add_epi32(s);
    const double ops = (double)nchain * iters;
    return ops / (t1 - t0);
}

/* A mix approximating the decoder: permute + shift + permute + fmadd per unit */
static double bench_mix(int nchain, int iters) {
    __m512i a[16];
    __m512 f[16];
    for (int i = 0; i < nchain; ++i) { a[i] = _mm512_set1_epi32(i + 1); f[i] = _mm512_set1_ps(1.0f); }
    const __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    const __m512i lut = _mm512_set1_epi32(0x3F800000);
    const __m512i shf = _mm512_set1_epi32(2);
    const __m512 vd = _mm512_set1_ps(1.0f);
    const double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < nchain; ++i) {
            const __m512i b = _mm512_permutevar_epi32(idx, a[i]);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            const __m512  g = _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut));
            f[i] = _mm512_fmadd_ps(g, vd, f[i]);
        }
    }
    const double t1 = now_s();
    __m512 s = _mm512_setzero_ps();
    for (int i = 0; i < nchain; ++i) s = _mm512_add_ps(s, f[i]);
    fsink = _mm512_reduce_add_ps(s);
    const double ops = 4.0 * nchain * iters;    /* 4 vector ops per chain unit */
    return ops / (t1 - t0);
}

static void report(const char *name, double (*fn)(int,int), int iters, int nthreads) {
    printf("  %-10s threads=%d\n", name, nthreads);
    printf("    %6s %14s %12s %12s\n", "chains", "vector ops/s", "ops/cycle", "thrpt/thread");
    for (int nc = 1; nc <= 8; nc *= 2) {
        double best = 0;
        double per_thread = 0;
        #pragma omp parallel num_threads(nthreads) reduction(max:best)
        {
            const double r = fn(nc, iters);
            if (r > best) best = r;
        }
        const double best_all = best * nthreads;   /* all threads did the same work */
        printf("    %6d %14.3fG %12.2f %12.3fG\n",
               nc, best_all / 1e9, best_all / (MHZ * 1e6), best / 1e9);
        per_thread = best;
        (void)per_thread;
    }
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : ITERS;
    printf("=== KNC vector issue-rate probe ===\n");
    printf("clock %.3f MHz, procs=%d, iters=%d\n\n", MHZ, omp_get_num_procs(), iters);

    for (int nt = 1; nt <= 4; nt *= 2) {
        report("vpaddd", bench_add, iters, nt);
    }
    for (int nt = 1; nt <= 4; nt *= 2) {
        report("vpermd", bench_perm, iters / 4, nt);
    }
    for (int nt = 1; nt <= 4; nt *= 2) {
        report("mix(4op)", bench_mix, iters / 4, nt);
    }

    printf("\nInterpretation:\n");
    printf("  ops/cycle is total vector ops per cycle, all threads summed.\n");
    printf("  2.00 = the core dual-issues; 1.00 = single-issue for this op class.\n");
    printf("  If ops/cycle does NOT rise with 'chains', the core is already\n");
    printf("  saturated and instruction COUNT is the only lever.\n");
    printf("  If ops/cycle rises with threads but not chains, the limit is\n");
    printf("  per-thread ILP -> more threads per core, or restructure the loop.\n");
    printf("\nsink %.3f  %d\n", (float)fsink, (int)isink);
    return 0;
}
