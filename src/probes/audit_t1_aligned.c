/* audit_t1_aligned.c -- host audit of aligned_f32()/aligned_f32_free() (model.c:120-131).
 *
 * Build (see report):
 *   zig cc -O1 -std=c11 -o D:\zigtest\t1.exe audit_t1_aligned.c gguf.c -I . -lm
 *   zig cc -O1 -std=c11 -fsanitize=address -o D:\zigtest\t1a.exe audit_t1_aligned.c gguf.c -I . -lm
 *
 * Questions answered by measurement:
 *   - is the returned pointer always 64-byte aligned?
 *   - is the usable region ever smaller than the requested count*4 bytes?
 *   - is p[-1] (the stored original malloc pointer) always inside the block,
 *     i.e. can the bookkeeping store ever land before the allocation?
 *   - is every byte of the requested region really writable? (ASan build)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "model.c"          /* to reach the file-static helpers */

int main(void)
{
    static const size_t counts[] = { 1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65,
                                     127, 128, 1000, 4096, 17408 };
    const size_t ncount = sizeof counts / sizeof counts[0];
    int raw_mod[64];
    memset(raw_mod, 0, sizeof raw_mod);

    long runs = 0, bad_align = 0, short_region = 0, book_before_raw = 0, bad_data = 0;
    size_t min_extra = (size_t)-1;      /* usable bytes beyond the requested ones */

    for (int rep = 0; rep < 64; ++rep) {
        for (size_t ci = 0; ci < ncount; ++ci) {
            const size_t count = counts[ci];
            const size_t want  = count * sizeof(float);
            float *p = aligned_f32(count, 64);
            if (!p) { printf("FAIL: aligned_f32 returned NULL\n"); return 2; }

            void *raw = ((void **)p)[-1];          /* same read as aligned_f32_free */
            runs++;

            if (((uintptr_t)p & 63u) != 0) { bad_align++; }

            /* The bookkeeping slot occupies [p-8,p); it must be inside malloc's block. */
            if ((uintptr_t)(p - 2) < (uintptr_t)raw) { book_before_raw++; }

            /* malloc'd size, from the source: count*4 + 64. Bytes usable from p: */
            const uintptr_t alloc_end = (uintptr_t)raw + want + 64;
            const uintptr_t usable    = alloc_end - (uintptr_t)p;
            if (usable < want) short_region++;
            if (usable - want < min_extra) min_extra = usable - want;
            raw_mod[(uintptr_t)raw & 63u]++;

            /* touch every requested element, then verify */
            for (size_t i = 0; i < count; ++i) p[i] = (float)i * 1.5f + 0.25f;
            for (size_t i = 0; i < count; ++i) {
                if (p[i] != (float)i * 1.5f + 0.25f) { bad_data++; break; }
            }
            aligned_f32_free(p);
        }
    }

    printf("T1 aligned_f32(count,64): %ld runs\n", runs);
    printf("   misaligned results            : %ld\n", bad_align);
    printf("   usable < requested bytes      : %ld\n", short_region);
    printf("   p[-1] slot before malloc block: %ld\n", book_before_raw);
    printf("   data write/read mismatches    : %ld\n", bad_data);
    printf("   minimum extra bytes beyond request: %zu (= raw %% 64)\n", min_extra);
    printf("   malloc raw %% 64 histogram:");
    for (int m = 0; m < 64; ++m) if (raw_mod[m]) printf(" %d:%d", m, raw_mod[m]);
    printf("\n");

    /* same with align=16, to show the arithmetic's dependence on the malloc
     * alignment being >= sizeof(void*) */
    long bad16 = 0;
    for (int rep = 0; rep < 32; ++rep) {
        float *p = aligned_f32(100, 16);
        if (((uintptr_t)p & 15u) != 0) bad16++;
        if ((uintptr_t)(p - 2) < (uintptr_t)((void **)p)[-1]) bad16 += 100;
        aligned_f32_free(p);
    }
    printf("   align=16 failures: %ld\n", bad16);

    printf("T1 %s\n", (bad_align || short_region || book_before_raw || bad_data || bad16)
                      ? "RESULT: DEFECT FOUND" : "RESULT: all properties hold");
    return 0;
}
