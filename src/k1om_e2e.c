/* k1om end-to-end acceptance: load the real 7.2 GB model on the card and run
 * the exact same greedy continuation that test_forward.c checks on the host.
 *
 * Why this exists: test_forward.c is a HOST harness (it needs the debug_tap
 * machinery and the ground-truth constants). This program is the minimal k1om
 * counterpart -- it exercises the real forward path end to end and asserts the
 * one thing that is checkable without a dump on the card: the official next
 * token for this prompt is 11751 (' Paris').
 *
 * A caveat that must not be forgotten: top-1 alone is NOT a correctness proof.
 * It was demonstrated that an entire RoPE-pairing change leaves top-1 at 11751.
 * The hidden-state check lives in test_forward.c on the host; this program is a
 * smoke test that the card build produces the same answer, not a full gate.
 *
 * Usage: k1om_e2e <model.gguf> [threads]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gguf.h"
#include "model.h"

static const int PROMPT[] = { 760, 6511, 314, 9338, 369 };  /* "The capital of France is" */
#define N_PROMPT 5
#define OFFICIAL_NEXT 11751                                 /* ' Paris' */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [threads]\n", argv[0]); return 2; }
    const char *path = argv[1];
    const int ctx = 64;

    printf("=== bonsai k1om end-to-end acceptance ===\n");
    printf("model : %s\n", path);
    fflush(stdout);

    double t0 = now_s();
    bonsai_model *m = bonsai_model_open(path, ctx);
    double t1 = now_s();
    if (!m) { fprintf(stderr, "FAIL: bonsai_model_open returned NULL\n"); return 1; }
    printf("load  : %.1f s\n", t1 - t0);
    bonsai_model_describe(m);
    fflush(stdout);

    bonsai_reset(m);
    int failed_at = -1;
    t0 = now_s();
    for (int i = 0; i < N_PROMPT; ++i) {
        const int rc = bonsai_forward(m, PROMPT[i], i);
        if (rc != 0) {
            fprintf(stderr, "FAIL: bonsai_forward(token %d, pos %d) returned %d\n",
                    PROMPT[i], i, rc);
            failed_at = i;
            break;
        }
    }
    t1 = now_s();

    int rc_out = 0;
    if (failed_at < 0) {
        const float *lg = m->sc.logits;
        int best = 0;
        for (int v = 1; v < m->n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        int rank = 0;
        for (int v = 0; v < m->n_vocab; ++v) if (lg[v] > lg[OFFICIAL_NEXT]) ++rank;

        printf("\n=== result ===\n");
        printf("  prefill : %d tokens in %.2f s  (%.3f tok/s)\n",
               N_PROMPT, t1 - t0, N_PROMPT / (t1 - t0));
        printf("  top-1   : %d   official %d (' Paris')   %s\n",
               best, OFFICIAL_NEXT, best == OFFICIAL_NEXT ? "MATCH" : "DIFFERENT");
        printf("  rank    : %d / %d\n", rank, m->n_vocab);
        printf("  [top-1 alone is NOT a correctness proof -- see test_forward.c]\n");

        if (best != OFFICIAL_NEXT) { printf("  FAIL: wrong next token\n"); rc_out = 1; }
        else {
            /* a few decode steps, timed, to catch a forward that works once
             * and then corrupts state */
            const int NDEC = 3;
            double d0 = now_s();
            for (int i = 0; i < NDEC; ++i) {
                const int r = bonsai_forward(m, best, N_PROMPT + i);
                if (r != 0) {
                    fprintf(stderr, "  FAIL: decode step %d returned %d\n", i, r);
                    rc_out = 1; break;
                }
            }
            double d1 = now_s();
            if (rc_out == 0)
                printf("  decode  : %.3f tok/s (%d tokens in %.2f s)\n",
                       NDEC / (d1 - d0), NDEC, d1 - d0);
        }
    } else {
        rc_out = 1;
    }

    bonsai_model_close(m);
    printf("\n%s\n", rc_out == 0 ? "k1om E2E PASS" : "k1om E2E FAIL");
    return rc_out;
}
