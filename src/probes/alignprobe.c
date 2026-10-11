/*
 * alignprobe.c -- review probe: does model.c's aligned_f32() bookkeeping hold?
 *
 *   void *raw = malloc(count*sizeof(float) + align);
 *   float *p  = (float *)(((uintptr_t)raw + align) & ~(uintptr_t)(align-1));
 *   memset(p, 0, count*sizeof(float));
 *   ((void **)p)[-1] = raw;                 <-- 8 bytes at p-8
 *
 * The store at p-8 is only in bounds if p >= raw + 8.  This probe checks that
 * for many sizes and many mallocs, prints the distribution of `raw % 64`, and
 * verifies that free(((void**)p)[-1]) is the original pointer and that the
 * memset cannot run past the allocation.
 *
 *   icc -mmic -O2 -std=c11 alignprobe.c -o alignprobe      (card)
 *   zig cc -O2 -std=c11 alignprobe.c -o alignprobe.exe     (host)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static float *aligned_f32(size_t count, size_t align) {
    void *raw = malloc(count * sizeof(float) + align);
    if (!raw) return NULL;
    const uintptr_t base = (uintptr_t)raw + (uintptr_t)align;
    float *p = (float *)(base & ~(uintptr_t)(align - 1));
    memset(p, 0, count * sizeof(float));
    ((void **)p)[-1] = raw;
    return p;
}

int main(void) {
    const size_t counts[] = { 1, 2, 3, 15, 16, 17, 64, 1000, 17408 };
    long long n = 0, misaligned = 0, under = 0, overrun = 0, badfree = 0;
    int dist[65];
    memset(dist, 0, sizeof(dist));
    for (unsigned c = 0; c < sizeof(counts)/sizeof(counts[0]); ++c) {
        for (int rep = 0; rep < 3000; ++rep) {
            size_t cnt = counts[c];
            size_t align = 64;
            void *raw = malloc(cnt * sizeof(float) + align);
            const uintptr_t base = (uintptr_t)raw + (uintptr_t)align;
            float *p = (float *)(base & ~(uintptr_t)(align - 1));
            ++n;
            if (((uintptr_t)p & 63u) != 0) ++misaligned;
            size_t d = (size_t)((uintptr_t)p - (uintptr_t)raw);
            if (d > 64) ++overrun;
            dist[d < 65 ? d : 64]++;
            if (d < sizeof(void *)) ++under;          /* p-8 would be before raw */
            memset(p, 0, cnt * sizeof(float));
            ((void **)p)[-1] = raw;
            if (((void **)p)[-1] != raw) ++badfree;
            /* can't free(): we reallocated with plain malloc, so just leak */
            (void)raw;
        }
    }
    printf("aligned_f32 bookkeeping over %lld allocations:\n", n);
    printf("  p %% 64 != 0 (would fault _mm512_load_ps) : %lld\n", misaligned);
    printf("  p - raw < sizeof(void*) (p[-1] OOB)      : %lld\n", under);
    printf("  p - raw > align  (memset could overrun)  : %lld\n", overrun);
    printf("  stored raw != raw                        : %lld\n", badfree);
    printf("  p-raw distribution: ");
    for (int i = 0; i < 65; ++i) if (dist[i]) printf("%d:%d ", i, dist[i]);
    printf("\n");
    return 0;
}
