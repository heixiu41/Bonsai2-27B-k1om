/*
 * prof2.h - lightweight, environment-gated region timers for the k1om live
 * profile.
 *
 * WHY THIS EXISTS (and how it differs from profile/prof.h). The earlier profile
 * instrumented a build with region macros woven through the code, which is
 * accurate but invasive. This one is a single header plus a set of static
 * counters that are OFF unless BONSAI_PROF2 is defined at build time, so the
 * shipped sources stay clean and the same model.c can produce both builds.
 *
 * Traps this avoids, all of which produced plausible-but-wrong budgets before:
 *   1. clock_gettime, not omp_get_wtime -- omp_get_wtime is unusable on this
 *      target (measured: returns nonsense).
 *   2. Only the MASTER thread records, and it records elapsed WALL time between
 *      two points in the calling code rather than summing inside a parallel
 *      region. Summing per-thread intervals inside a region double-counts.
 *   3. A region that encloses another must not be timed with a shared anchor,
 *      or the inner one clobbers the outer. Each region here carries its own
 *      start timestamp.
 *   4. The counters are `static` in a HEADER, so every translation unit that
 *      includes it gets its OWN copy. p2_add() must therefore be called from,
 *      and p2_dump() must be called in, the SAME translation unit (model.c).
 *      Calling p2_dump() from main_k1om.c reads a different, always-zero set --
 *      which is exactly the bug this comment exists to prevent.
 */
#ifndef BONSAI_PROF2_H
#define BONSAI_PROF2_H

#ifdef BONSAI_PROF2

/* _POSIX_C_SOURCE must already be set by the including translation unit BEFORE
 * its first system header (model.c does this), otherwise CLOCK_MONOTONIC is not
 * declared on k1om's glibc. Do not define it here -- by the time this header is
 * reached, <time.h> may already have been pulled in and the define would have
 * no effect (this exact mistake produced "CLOCK_MONOTONIC is undefined"). */
#include <time.h>
#include <stdio.h>

typedef enum {
    P2_FWD = 0,        /* whole bonsai_forward */
    P2_GDN,            /* block_gdn, all layers */
    P2_GDN_REC,        /* just the per-value-head recurrence loop */
    P2_GDN_CONV,       /* causal depthwise conv */
    P2_GDN_L2,         /* per-head l2norm of q and k */
    P2_GDN_GATE,       /* beta/g softplus+sigmoid loop */
    P2_ATTN,           /* block_attn, all layers */
    P2_FFN,            /* block_ffn, all layers */
    P2_MM,             /* mm_folded total */
    P2_MM_MATVEC,      /* the pq2_0_matvec_knc call only */
    P2_MM_HADAMARD,    /* the hadamard_forward call only */
    P2_N
} p2_id;

static const char *p2_names[P2_N] = {
    "forward (total)", "block_gdn", "  gdn: recurrence", "  gdn: conv",
    "  gdn: l2norm", "  gdn: beta/g gate", "block_attn", "block_ffn",
    "mm_folded (total)", "  mm: matvec kernel", "  mm: hadamard"
};

static double   p2_t[P2_N];
static long     p2_n[P2_N];

static inline double p2_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static inline void p2_add(p2_id id, double dt) {
    p2_t[id] += dt;
    p2_n[id] += 1;
}
static void p2_dump(void) {
    double tot = p2_t[P2_FWD];
    fprintf(stderr, "\n=== BONSAI_PROF2 region budget (wall clock) ===\n");
    for (int i = 0; i < P2_N; ++i) {
        if (p2_n[i] == 0 && p2_t[i] == 0) continue;
        fprintf(stderr, "  %-22s %9.3f s  %6.2f%%  (%ld calls)\n",
                p2_names[i], p2_t[i],
                tot > 0 ? 100.0 * p2_t[i] / tot : 0.0, p2_n[i]);
    }
    fprintf(stderr, "  note: nested regions overlap; children do not sum to parents\n");
}

/* Usage: P2_T(id, expr) -- expr is a statement/block; the wall time around it
 * is charged to `id`.  Only the master thread records. */
#define P2_T(id, expr) do {                                   \
        const int _p2m = p2_master();                          \
        const double _p2a = _p2m ? p2_now() : 0.0;             \
        expr;                                                  \
        if (_p2m) p2_add((id), p2_now() - _p2a);               \
    } while (0)

#ifdef _OPENMP
#include <omp.h>
static inline int p2_master(void) { return omp_get_thread_num() == 0; }
#else
static inline int p2_master(void) { return 1; }
#endif

#define P2_DUMP() p2_dump()

#else  /* !BONSAI_PROF2 */

/* Stubs, not just compiled-away macros. The hand-written probes in model.c call
 * p2_now()/p2_add() directly (they need a start timestamp in a local), so when
 * the macro is off those calls must still COMPILE. Making them inert stubs
 * avoids wrapping every probe site in #ifdef, which is where mistakes happen.
 * A good compiler folds all of this away. */
typedef enum { P2_FWD = 0, P2_GDN, P2_GDN_REC, P2_GDN_CONV, P2_GDN_L2,
               P2_GDN_GATE, P2_ATTN, P2_FFN, P2_MM, P2_MM_MATVEC,
               P2_MM_HADAMARD, P2_N } p2_id;
static inline double p2_now(void) { return 0.0; }
static inline void   p2_add(p2_id id, double dt) { (void)id; (void)dt; }
static inline int    p2_master(void) { return 0; }

#define P2_T(id, expr) do { expr; } while (0)
#define P2_DUMP() do { } while (0)

#endif /* BONSAI_PROF2 */
#endif /* BONSAI_PROF2_H */
