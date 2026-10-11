/* rma_probe_sink.c -- card-side (k1om) feasibility probe for the SCIF RMA
 * direct-transfer path.
 *
 * Answers, on real hardware, the questions the live design depends on:
 *   1. does scif_register() work from a card user process, how long does it
 *      take, and what offset does it hand back?
 *   2. do host scif_writeto() RMAs actually land in the registered window,
 *      and is the card CPU able to observe them (L1/L2 coherency)?
 *   3. what raw bandwidth does the RMA path give at each chunk size?
 *   4. does the host's scif_fence_signal(SCIF_SIGNAL_REMOTE) write reach the
 *      card's memory?
 *
 * Protocol (port 20501):
 *   host -> card : uint64 win_size
 *   card -> host : struct probe_rep { int64 rma_off; uint64 win; uint64 reg_us; }
 *   then, repeatedly:
 *     host -> card : uint64 cmd     (0 = checksum, 1 = report ctrl tail, 2 = quit)
 *     card -> host : uint64 r[4]
 *
 * Build: icc -mmic -O2 -std=c11 rma_probe_sink.c -o rma_probe_sink -lscif
 * Run:   ./rma_probe_sink <win_bytes>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <scif.h>

#include "scif_rma.h"

struct probe_rep {
    int64_t  rma_off;
    uint64_t win;
    uint64_t reg_us;
};

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int recv_full(scif_epd_t e, void *buf, int n) {
    int got = 0;
    while (got < n) {
        int rc = scif_recv(e, (char *)buf + got, n - got, SCIF_RECV_BLOCK);
        if (rc <= 0) { perror("scif_recv"); return -1; }
        got += rc;
    }
    return got;
}

static int send_full(scif_epd_t e, const void *buf, int n) {
    int sent = 0;
    while (sent < n) {
        int rc = scif_send(e, (char *)buf + sent, n - sent, SCIF_SEND_BLOCK);
        if (rc <= 0) { perror("scif_send"); return -1; }
        sent += rc;
    }
    return sent;
}

int main(int argc, char **argv) {
    size_t win = 512ULL << 20;
    if (argc > 1) win = (size_t)strtoull(argv[1], NULL, 0);
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) { fprintf(stderr, "sysconf(_SC_PAGESIZE) failed\n"); return 1; }
    win = (win + (size_t)ps - 1) & ~((size_t)ps - 1);

    scif_epd_t lepd = scif_open();
    if (lepd == SCIF_OPEN_FAILED) { perror("scif_open"); return 1; }
    const int port = scif_bind(lepd, SCIF_RMA_PROBE_PORT);
    if (port < 0) { perror("scif_bind"); return 1; }
    if (scif_listen(lepd, 1) < 0) { perror("scif_listen"); return 1; }
    printf("rma_probe_sink: listening on %d (page=%ld, win=%zu)\n",
           port, ps, win);
    fflush(stdout);

    struct scif_portID peer;
    scif_epd_t epd;
    if (scif_accept(lepd, &peer, &epd, SCIF_ACCEPT_SYNC) < 0) {
        perror("scif_accept");
        return 1;
    }
    printf("rma_probe_sink: connection from node %d\n", peer.node);
    fflush(stdout);

    void *buf = NULL;
    if (posix_memalign(&buf, (size_t)ps, win) != 0) {
        perror("posix_memalign");
        return 1;
    }
    /* Poison the window *and* pull it into the caches: if the card still sees
     * 0xA5 after the host's RMAs, the DMA is either not landing or the caches
     * are stale -- both are things the live design must know about. */
    memset(buf, 0xA5, win);

    const double t_reg0 = now_s();
    const off_t off = scif_register(epd, buf, win, 0,
                                    SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    const double t_reg1 = now_s();
    if (off == SCIF_REGISTER_FAILED) {
        perror("scif_register");
        return 1;
    }
    printf("rma_probe_sink: registered %zu bytes at offset %lld (%.3f s)\n",
           win, (long long)off, t_reg1 - t_reg0);
    fflush(stdout);

    struct probe_rep rep;
    rep.rma_off = (int64_t)off;
    rep.win = (uint64_t)win;
    rep.reg_us = (uint64_t)((t_reg1 - t_reg0) * 1e6);
    if (send_full(epd, &rep, (int)sizeof rep) < 0) return 1;

    for (;;) {
        uint64_t cmd = 0;
        if (recv_full(epd, &cmd, 8) < 0) break;

        uint64_t r[4] = { 0, 0, 0, 0 };
        if (cmd == 0) {
            /* Full-window integrity report against the known pattern. */
            const uint64_t *w = (const uint64_t *)buf;
            const uint64_t nwords = (uint64_t)(win / 8);
            uint64_t n_stale = 0, n_bad = 0, first_bad = (uint64_t)-1;
            const double t0 = now_s();
            for (uint64_t i = 0; i < nwords; ++i) {
                const uint64_t v = w[i];
                if (v == 0xA5A5A5A5A5A5A5A5ULL) ++n_stale;
                else if (v != scif_rma_pattern(i)) {
                    ++n_bad;
                    if (first_bad == (uint64_t)-1) first_bad = i;
                }
            }
            r[0] = scif_rma_ck(SCIF_RMA_CK_INIT, buf, win);
            r[1] = n_stale;
            r[2] = n_bad;
            r[3] = first_bad;
            printf("rma_probe_sink: verify %.1f s  stale=%llu bad=%llu "
                   "first_bad=%lld ck=%016llx\n",
                   now_s() - t0,
                   (unsigned long long)n_stale, (unsigned long long)n_bad,
                   (long long)(int64_t)first_bad,
                   (unsigned long long)r[0]);
            fflush(stdout);
        } else if (cmd == 1) {
            /* The host pokes a magic value into the last 8 bytes of the window
             * with scif_fence_signal(SCIF_SIGNAL_REMOTE); report what we see. */
            uint64_t tail = 0;
            memcpy(&tail, (const char *)buf + win - 8, 8);
            r[0] = tail;
            printf("rma_probe_sink: tail = 0x%016llx\n",
                   (unsigned long long)tail);
            fflush(stdout);
        } else if (cmd == 3) {
            /* Can this core actually observe a control word that the host
             * updates by RMA while we spin on it?  KNC's L1 is not coherent,
             * so this is the one mechanism the live sink depends on that the
             * batch checksum below cannot prove.  The host sleeps 300 ms, sets
             * the word to 1, sleeps 100 ms, then sets it to 2. */
            const volatile uint64_t *vp =
                (const volatile uint64_t *)((const char *)buf + win - 16);
            uint64_t seen1 = 0, seen2 = 0;
            double w1 = 0, w2 = 0;
            double t0 = now_s();
            while (now_s() - t0 < 5.0)
                if (*vp == 1ULL) { seen1 = 1; w1 = now_s() - t0; break; }
            t0 = now_s();
            while (now_s() - t0 < 5.0)
                if (*vp == 2ULL) { seen2 = 1; w2 = now_s() - t0; break; }
            r[0] = seen1;
            r[1] = seen2;
            r[2] = (uint64_t)(w1 * 1e6);
            r[3] = (uint64_t)(w2 * 1e6);
            printf("rma_probe_sink: poll test saw1=%llu saw2=%llu "
                   "wait1_us=%llu wait2_us=%llu\n",
                   (unsigned long long)seen1, (unsigned long long)seen2,
                   (unsigned long long)r[2], (unsigned long long)r[3]);
            fflush(stdout);
        } else if (cmd == 4) {
            /* Card writes a magic into the window tail so the host can read it
             * back with scif_readfrom (the card -> host RMA direction). */
            const uint64_t m = 0x1122334455667788ULL;
            memcpy((char *)buf + win - 8, &m, 8);
            r[0] = m;
            printf("rma_probe_sink: tail magic written\n");
            fflush(stdout);
        } else {
            break;
        }
        if (send_full(epd, r, (int)sizeof r) < 0) break;
    }

    printf("rma_probe_sink: closing\n");
    fflush(stdout);
    scif_unregister(epd, off, win);
    scif_close(epd);
    scif_close(lepd);
    free(buf);
    return 0;
}
