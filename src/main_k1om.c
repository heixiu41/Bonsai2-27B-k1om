/* k1om (Xeon Phi Knights Corner) entry point for the Ternary Bonsai 2 27B
 * inference engine.
 *
 * Usage:
 *   bonsai_k1om <model.gguf> [selftest|forward|chat] [threads]
 *
 *   mode    "selftest" (default) = load the GGUF header/table and validate it,
 *           "forward"            = full load + a 5-token prefill, a greedy
 *                                  continuation and a throughput report.
 *           Anything else is rejected with the usage above.
 *   threads OpenMP threads. Measured fastest on this 7120P after the serial
 *           glue in block_gdn() and hadamard_forward() was parallelised
 *           (docs/K1OM_TIME_BUDGET.md): 244 (1.634 tok/s) > 183 (1.490) >
 *           122 (1.396) > 61 (1.009). Before those fixes the optimum was 122,
 *           because the run was dominated by serial work that no thread count
 *           could touch. 244 is now the default when neither this argument nor
 *           the OMP_NUM_THREADS environment variable is set.
 *           Ignored when built without OpenMP.
 *
 * Exit code: 0 only if the requested mode fully succeeded.
 *
 * These two things have never actually executed on the coprocessor before:
 *   - pq2_0.h's IMCI kernels (_mm512_*) -- they only assemble for k1om
 *   - the gguf loader's 64-bit file offsets over a 7.2 GB file
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>     /* unlink (chat mode frees the tmpfs model copy) */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gguf.h"
#include "model.h"

#define DEFAULT_THREADS 244

double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s <model.gguf> [selftest|forward|chat] [threads]\n",
            argv0);
}

/* ---- chat mode: persistent line-based token server --------------------
 * Protocol (stdin -> stdout, one line each, stdout line-buffered):
 *   E <id>      set the EOS token id (generation stops after emitting it)
 *   R           reset all recurrent state and position; replies "OK"
 *   P <id...>   ingest prompt tokens at consecutive positions; replies "OK"
 *   G <max>     greedy-generate up to <max> tokens; prints one "<id>" line
 *               per token as produced, then "END <n>". Also stops at EOS
 *               (if set) or when the context is full.
 * Unknown lines get "ERR <line>". The model loads ONCE at startup; the
 * caller keeps the pipe open for the whole chat session. */
#define CHAT_CTX 8192
/* Prefill only needs the logits of its LAST token, so the earlier prompt
 * tokens skip the n_vocab-wide output head.  BONSAI_NO_PREFILL_SKIP=1 restores
 * the old "compute it every token" behaviour; that switch is how the A/B was
 * run (one binary, both orders). */
static int prefill_skip_head(void) {
    static int r = -1;
    if (r < 0) r = (getenv("BONSAI_NO_PREFILL_SKIP") == NULL);
    return r;
}
static int chat_mode(bonsai_model *m) {
    char line[65536];
    int32_t pos = 0;
    int32_t eos = -1;
    int32_t ntok_max = 1024 * 1024;      /* sanity bound per command */

    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("READY\n");
    while (fgets(line, sizeof(line), stdin)) {
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
        if (*p == '\n' || *p == '\0') continue;
        const char cmd = *p++;
        if (cmd == 'E') {
            eos = (int32_t)strtol(p, NULL, 10);
            printf("OK\n");
        } else if (cmd == 'R') {
            bonsai_reset(m);
            pos = 0;
            printf("OK\n");
        } else if (cmd == 'P') {
            /* NB: a Windows client in text mode sends \r\n -- treat \r as
             * whitespace/terminator or the last token fails strtol.
             *
             * One token of lookahead: only the LAST prompt token needs logits
             * (the first G reads them), so every earlier token goes through
             * bonsai_forward_nologits and skips the n_vocab-wide output head
             * -- the whole prefill is read-out-free until its final token. */
            int fail = 0, have = 0;
            int32_t pend = 0;
            const int skip = prefill_skip_head();
            char *end = p;
            for (;;) {
                while (*end == ' ' || *end == '\t' || *end == '\r') ++end;
                if (*end == '\n' || *end == '\0') break;
                char *next = NULL;
                const long v = strtol(end, &next, 10);
                if (next == end || v < 0 || v > ntok_max) { fail = 1; break; }
                end = next;
                if (have) {
                    if (pos >= m->ctx) { fail = 2; break; }
                    const int rc = skip ? bonsai_forward_nologits(m, pend, pos)
                                        : bonsai_forward(m, pend, pos);
                    ++pos;
                    if (rc != 0) { fail = 2; break; }
                }
                pend = (int32_t)v;
                have = 1;
            }
            if (!fail && have) {
                if (pos >= m->ctx ||
                    bonsai_forward(m, pend, pos++) != 0) fail = 2;
            }
            printf(fail ? "ERR prompt (code %d)\n" : "OK\n", fail);
        } else if (cmd == 'G') {
            const long max = strtol(p, NULL, 10);
            long made = 0;
            while (made < max && pos < m->ctx) {
                /* greedy argmax over the logits of the last forward */
                const int32_t best = bonsai_argmax(m->sc.logits, m->n_vocab);
                printf("%d\n", best);
                ++made;
                if (best == eos) break;
                if (bonsai_forward(m, best, pos++) != 0) break;
            }
            printf("END %ld\n", made);
        } else {
            printf("ERR %.40s\n", line);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const char *path = argv[1];
    const char *mode = argc > 2 ? argv[2] : "selftest";

    /* Whitelist, not blacklist: a typo must not silently run the 7.2 GB load
     * and the full forward path. */
    if (strcmp(mode, "selftest") != 0 && strcmp(mode, "forward") != 0 &&
        strcmp(mode, "chat") != 0) {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        usage(argv[0]);
        return 2;
    }

    int threads = DEFAULT_THREADS;
    const int threads_given = (argc > 3);
    if (threads_given) {
        char *end = NULL;
        long v = strtol(argv[3], &end, 10);
        if (!end || *end != '\0' || v < 1 || v > 4096) {
            fprintf(stderr, "invalid thread count '%s'\n", argv[3]);
            usage(argv[0]);
            return 2;
        }
        threads = (int)v;
    }
    if (argc > 4) { usage(argv[0]); return 2; }

    /* Precedence: explicit argument > OMP_NUM_THREADS > measured default. */
    const char *thread_src = "default";
#ifdef _OPENMP
    if (threads_given) {
        omp_set_num_threads(threads);
        thread_src = "argument";
    } else if (getenv("OMP_NUM_THREADS")) {
        thread_src = "OMP_NUM_THREADS";
    } else {
        omp_set_num_threads(threads);
    }
    threads = omp_get_max_threads();
#else
    (void)threads_given;
    thread_src = "ignored (built without OpenMP)";
#endif

    if (strcmp(mode, "chat") != 0) {
        /* chat mode keeps stdout protocol-clean (banner would confuse the
         * line parser); its diagnostics go to stderr instead. */
        printf("=== bonsai k1om ===\n");
        printf("model  : %s\n", path);
        printf("mode   : %s\n", mode);
        printf("threads: %d (%s)\n", threads, thread_src);
        fflush(stdout);
    }

#ifdef __MIC__
    /* libiomp5 reserves ~117 MB of VIRTUAL stack per thread on KNC
     * (244 threads -> ~28 GB VmSize). Mostly never touched, but it makes
     * every fork() a fork-bomb on a 15.5 GB card and inflates the OOM
     * score. Our deepest per-thread frame is a few KB; 16 MB is lavish. */
    kmp_set_stacksize_s(16 * 1024 * 1024);
#endif

    if (strcmp(mode, "selftest") == 0) {
        double t0 = now_s();
        int rc = bonsai_gguf_selftest(path);
        double t1 = now_s();
        printf("\nselftest rc=%d  elapsed %.2fs\n", rc, t1 - t0);
        return rc == 0 ? 0 : 1;
    }

    /* ---- persistent chat server (protocol: see chat_mode comment) ---- */
    if (strcmp(mode, "chat") == 0) {
        const double t0 = now_s();
        bonsai_model *m = bonsai_model_open(path, CHAT_CTX);
        if (!m) {
            fprintf(stderr, "failed to open model '%s'\n", path);
            return 1;
        }
        /* load diagnostics go to stderr so stdout carries only protocol */
        fprintf(stderr, "chat: model loaded in %.1fs, ctx=%d\n",
                now_s() - t0, CHAT_CTX);
        /* The card's rootfs (incl. /tmp) is TMPFS with no swap: the 7.2 GB
         * model file sits in unevictable page cache WHILE we also hold the
         * full copy in anonymous memory -- 14.5 GB of 15.5 GB, and the
         * 2.6.38 OOM killer fires on the next spike (measured twice, e.g.
         * the first 1 GB kv memset of ctx=8192). The loader copies the whole
         * file into g->map and closes the fd (gguf.c), so after open the file
         * is dead weight: UNLINK it and the tmpfs pages are freed. Only chat
         * mode does this (one load per session; /tmp is volatile anyway). */
        if (unlink(path) == 0)
            fprintf(stderr, "chat: model file unlinked (freed ~7 GB tmpfs)\n");
        else
            fprintf(stderr, "chat: WARNING unlink(%s) failed, memory tight\n",
                    path);
        /* Pre-fault the ctx=8192 kv cache (1 GB) now that there is room. */
        bonsai_reset(m);
        fprintf(stderr, "chat: kv pre-faulted\n");
        const int rc = chat_mode(m);
        bonsai_model_close(m);
        return rc;
    }

    /* ---- full load + a short greedy continuation ---- */
    static const int PROMPT[] = { 760, 6511, 314, 9338, 369 };  /* "The capital of France is" */
    const int N_PROMPT = 5;
    const int OFFICIAL_NEXT = 11751;                            /* ' Paris' */

    const int32_t ctx = 256;
    double t0 = now_s();
    bonsai_model *m = bonsai_model_open(path, ctx);
    double t1 = now_s();
    if (!m) {
        fprintf(stderr, "failed to open model '%s' "
                        "(not found, truncated, or unsupported layout)\n", path);
        return 1;
    }
    printf("\nload time %.1fs\n", t1 - t0);
    bonsai_model_describe(m);
    fflush(stdout);

    /* Sticky failure flag: any failed forward makes the whole run fail, even
     * if the logits left behind happen to still predict the official token. */
    int failed = 0;

    bonsai_reset(m);
    t0 = now_s();
    for (int i = 0; i < N_PROMPT; ++i) {
        double p0 = now_s();
        if (bonsai_forward(m, PROMPT[i], i) != 0) {
            fprintf(stderr, "forward failed at prompt token %d\n", i);
            failed = 1;
            break;
        }
        printf("  prefill token %d: %.3fs\n", i, now_s() - p0);
    }
    t1 = now_s();

    int best = -1;
    if (!failed) {
        const float *lg = m->sc.logits;
        best = 0;
        for (int v = 1; v < m->n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        int rank = 0;
        for (int v = 0; v < m->n_vocab; ++v) if (lg[v] > lg[OFFICIAL_NEXT]) ++rank;

        printf("\n=== result ===\n");
        printf("  prompt %d tokens in %.2fs  (%.3f tok/s prefill)\n",
               N_PROMPT, t1 - t0, N_PROMPT / (t1 - t0));
        printf("  top-1 = %d   official = %d (' Paris')   %s\n",
               best, OFFICIAL_NEXT, best == OFFICIAL_NEXT ? "MATCH" : "DIFFERENT");
        printf("  rank of %d = %d / %d\n", OFFICIAL_NEXT, rank, m->n_vocab);
        printf("  NOTE: top-1/rank alone do not prove the forward pass; see\n"
               "        test_forward.c's hidden-state landmarks.\n");

        /* decode throughput on a few extra tokens */
        t0 = now_s();
        const int NDEC = 4;
        int done = 0;
        for (int i = 0; i < NDEC; ++i) {
            double d0 = now_s();
            if (bonsai_forward(m, best, N_PROMPT + i) != 0) {
                fprintf(stderr, "decode forward failed at step %d\n", i + 1);
                failed = 1;
                break;
            }
            printf("  decode token %d: %.3fs\n", i, now_s() - d0);
            ++done;
        }
        t1 = now_s();
        printf("  decode %.3f tok/s (%d/%d tokens in %.2fs)\n",
               done > 0 ? done / (t1 - t0) : 0.0, done, NDEC, t1 - t0);
    } else {
        printf("\n=== result ===\n  prefill failed; no result\n");
    }

    bonsai_model_close(m);
    if (failed) return 1;
    return best == OFFICIAL_NEXT ? 0 : 1;
}
