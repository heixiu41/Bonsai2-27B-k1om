/*
 * ggufguard.c -- review probe: does the GGUF loader read out of bounds on a
 * malformed / truncated file, or does it fail cleanly?
 *
 * Every allocation inside bonsai_gguf_load() comes back from a private
 * allocator that returns memory ending exactly at a PROT_NONE guard page, so
 * any read past the end of an allocation is a SIGSEGV instead of silence.
 * Each case runs in a forked child so one fault does not stop the sweep.
 *
 *   icc -mmic -O2 -std=c11 ggufguard.c gguf.c -o ggufguard -lm
 */
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gguf.h"

/* ---------------- guard-paged allocator (interposes libc) --------------- */
#define MAXREG 512
static struct { void *ret; void *base; size_t len; } reg[MAXREG];
static int nreg = 0;

static void *gp_alloc(size_t n) {
    if (n == 0) n = 1;
    size_t pages = (n + 4095) / 4096;
    size_t len = (pages + 1) * 4096;
    uint8_t *p = (uint8_t *)mmap(NULL, len, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    if (mprotect(p + pages * 4096, 4096, PROT_NONE) != 0) return NULL;
    void *ret = p + pages * 4096 - n;      /* ends exactly at the guard page */
    if (nreg < MAXREG) { reg[nreg].ret = ret; reg[nreg].base = p;
                         reg[nreg].len = len; ++nreg; }
    return ret;
}

void *malloc(size_t n) { return gp_alloc(n); }
void *calloc(size_t a, size_t b) {
    size_t n = a * b;
    void *p = gp_alloc(n);
    if (p) memset(p, 0, n);
    return p;
}
void free(void *p) {
    if (!p) return;
    for (int i = 0; i < nreg; ++i)
        if (reg[i].ret == p) { munmap(reg[i].base, reg[i].len); return; }
}
void *realloc(void *p, size_t n) {
    void *q = gp_alloc(n);
    if (q && p) memcpy(q, p, n);
    free(p);
    return q;
}

/* ------------------------------------------------------------------------ */
static void case_load(const char *path, int touch_tensors) {
    printf("  %-24s : ", path);
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        bonsai_gguf *g = bonsai_gguf_load(path);
        printf("%s", g ? "loaded" : "NULL(clean)");
        fflush(stdout);
        if (g && touch_tensors) {
            printf("  touching tensor data...");
            fflush(stdout);
            for (uint64_t i = 0; i < g->tensor_len; ++i) {
                const uint8_t *d = bonsai_tensor_data(g, &g->tensors[i]);
                volatile uint8_t v = d[0];
                (void)v;
            }
            printf(" ok");
        }
        fflush(stdout);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) printf("  << KILLED BY SIGNAL %d (out-of-bounds access)\n",
                                WTERMSIG(st));
    else                 printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/tmp";
    char buf[512];
    printf("=== GGUF loader under a guard page after every allocation ===\n");
    struct { const char *f; int touch; } cases[] = {
        { "min_valid.gguf",        0 },
        { "trunc_tensors.gguf",    0 },
        { "trunc_kv_scalar.gguf",  0 },
        { "trunc_array.gguf",      0 },
        { "trunc_signvalues.gguf", 0 },
        { "no_data.gguf",          1 },
        { "bad_offset.gguf",       1 },
        { "dims_overflow.gguf",    1 },
    };
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        snprintf(buf, sizeof(buf), "%s/%s", dir, cases[i].f);
        case_load(buf, cases[i].touch);
    }
    return 0;
}
