/* scif_rma_sink.c -- card-side (k1om) file receiver over SCIF RMA.
 *
 * Replacement for tools/scif_sink.c.  The *handshake* is byte-identical to the
 * legacy message sink (constraint 2): the host still opens port 20500 and
 * sends the same 8-byte LE size + 248-byte NUL-padded name.  Only the data
 * plane changed: instead of 16 MB scif_send/scif_recv messages the card
 * registers a ring window once and the host DMAs straight into it with
 * scif_writeto.
 *
 * Why the card must own the window: the card holds the model in tmpfs, so a
 * registered ring is the only way to receive without a copy through the SCIF
 * message path.  A 256 MB ring is used rather than the whole file because the
 * 7120P only has 15.5 GB of RAM and / is a 12.9 GiB tmpfs.
 *
 * Flow control (both directions ride the window itself -- no messages in the
 * hot path):
 *   host -> card  ctrl[RMA_CTRL_HOST_SEQ]: how many slots the host has written,
 *                 published with scif_fence_signal(SCIF_SIGNAL_REMOTE) so it
 *                 cannot land before the slot data it refers to.
 *   card -> host  ctrl[RMA_CTRL_CARD_SEQ]: how many slots have been copied to
 *                 the file, read back by the host with scif_readfrom.  The
 *                 host stays at most <nslots> ahead of this counter.
 *
 * The card flushes slots strictly in order, so slot k is safe to read as soon
 * as ctrl[RMA_CTRL_HOST_SEQ] > k.  Reading slot k+1 while the host is already
 * filling slot k+2 is what hides the write latency (the ring depth does the
 * double buffering).
 *
 * Build: icc -mmic -O2 -std=c11 scif_rma_sink.c -o scif_rma_sink -lscif
 * Run:   ./scif_rma_sink          (one transfer, then exit, like the legacy sink)
 * Tune:  SCIF_RMA_SLOT_MB (default 32), SCIF_RMA_SLOTS (default 8)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <scif.h>

#include "scif_rma.h"

#define DEFAULT_SLOT_MB 32
#define DEFAULT_SLOTS   8
#define STALL_TIMEOUT_S 300.0

/* keep the wire structs honest on this side too */
typedef char assert_hdr_size[(sizeof(struct scif_hdr) == 256) ? 1 : -1];
typedef char assert_geom_size[(sizeof(struct scif_rma_geom) == 48) ? 1 : -1];
typedef char assert_res_size[(sizeof(struct rma_result) == 64) ? 1 : -1];

/* Shared state for the overlapped checksum thread.  The flusher marks
 * ready[k] once slot k is complete in the window; the checksummer reads the
 * slot and sets ckdone[k]; only then does the flusher publish CARD_SEQ and
 * let the host reuse the slot.  Because the checksummer only ever reads the
 * window, it can run while the flusher is blocked in write(). */
struct ck_ctx {
    const char     *win;
    size_t          slot_base;
    size_t          slot;
    uint64_t        nslots;
    uint64_t        total_slots;
    uint64_t        size;
    volatile int   *ready;
    volatile int   *ckdone;
    volatile int    abort;
    volatile uint64_t ck;
};

static void *ck_worker(void *arg) {
    struct ck_ctx *c = (struct ck_ctx *)arg;
    uint64_t ck = SCIF_RMA_CK_INIT;
    uint64_t done_bytes = 0;
    for (uint64_t k = 0; k < c->total_slots; ++k) {
        while (!c->ready[k]) {
            if (c->abort) return NULL;
        }
        __sync_synchronize();
        const uint64_t remain = c->size - done_bytes;
        const size_t n = (size_t)((remain < c->slot) ? remain : c->slot);
        const size_t off = c->slot_base + (size_t)(k % c->nslots) * c->slot;
        ck = scif_rma_ck(ck, c->win + off, n);
        done_bytes += n;
        c->ck = ck;
        __sync_synchronize();
        c->ckdone[k] = 1;
    }
    return NULL;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int recv_full(scif_epd_t e, void *buf, int n) {
    int got = 0;
    while (got < n) {
        const int rc = scif_recv(e, (char *)buf + got, n - got, SCIF_RECV_BLOCK);
        if (rc <= 0) { perror("scif_recv"); return -1; }
        got += rc;
    }
    return got;
}

static int send_full(scif_epd_t e, const void *buf, int n) {
    int sent = 0;
    while (sent < n) {
        const int rc = scif_send(e, (char *)buf + sent, n - sent, SCIF_SEND_BLOCK);
        if (rc <= 0) { perror("scif_send"); return -1; }
        sent += rc;
    }
    return sent;
}

static uint64_t env_u64(const char *name, uint64_t dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    return (uint64_t)strtoull(v, NULL, 0);
}

int main(void) {
    const long ps_l = sysconf(_SC_PAGESIZE);
    if (ps_l <= 0) { fprintf(stderr, "sysconf(_SC_PAGESIZE) failed\n"); return 1; }
    const size_t ps = (size_t)ps_l;

    size_t slot = (size_t)env_u64("SCIF_RMA_SLOT_MB", DEFAULT_SLOT_MB) << 20;
    uint64_t nslots = env_u64("SCIF_RMA_SLOTS", DEFAULT_SLOTS);
    if (slot < ps) slot = ps;
    slot = (slot + ps - 1) & ~(ps - 1);
    if (nslots < 2) nslots = 2;
    if (nslots > 64) nslots = 64;

    /* Counters at the start of the window; the ring starts on the next page so
     * the two never overlap (an overlap silently feeds file bytes to the
     * credit checks). */
    const size_t ctrl_off = 0;
    const size_t slot_base =
        (RMA_CTRL_BYTES + (size_t)ps - 1) & ~((size_t)ps - 1);
    size_t win = slot_base + (size_t)nslots * slot;
    win = (win + (size_t)ps - 1) & ~((size_t)ps - 1);

    scif_epd_t lepd = scif_open();
    if (lepd == SCIF_OPEN_FAILED) { perror("scif_open"); return 1; }
    const int port = scif_bind(lepd, SCIF_RMA_PORT);
    if (port < 0) { perror("scif_bind"); return 1; }
    if (scif_listen(lepd, 1) < 0) { perror("scif_listen"); return 1; }
    printf("scif_rma_sink: listening on port %d (window %zu MiB = %llu x %zu MiB)\n",
           port, win >> 20, (unsigned long long)nslots, slot >> 20);
    fflush(stdout);

    struct scif_portID peer;
    scif_epd_t epd;
    if (scif_accept(lepd, &peer, &epd, SCIF_ACCEPT_SYNC) < 0) {
        perror("scif_accept");
        return 1;
    }
    printf("scif_rma_sink: connection from node %d\n", peer.node);
    fflush(stdout);

    struct scif_hdr h;
    memset(&h, 0, sizeof h);
    if (recv_full(epd, &h, (int)sizeof h) < 0) return 1;
    h.name[sizeof(h.name) - 1] = '\0';

    struct rma_result res;
    memset(&res, 0, sizeof res);

    char path[320];
    snprintf(path, sizeof path, "/tmp/%s", h.name);
    if (h.size == 0) {
        res.status = RMA_ST_ARGS;
        res.err = EINVAL;
        send_full(epd, &res, (int)sizeof res);
        return 1;
    }

    char *winbuf = NULL;
    if (posix_memalign((void **)&winbuf, ps, win) != 0) {
        perror("posix_memalign");
        res.status = RMA_ST_ALLOC;
        res.err = ENOMEM;
        send_full(epd, &res, (int)sizeof res);
        return 1;
    }
    memset(winbuf, 0, slot_base);               /* counters start at zero */

    const double t_reg0 = now_s();
    const off_t rma_off = scif_register(epd, winbuf, win, 0,
                                        SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    const double t_reg1 = now_s();
    if (rma_off == SCIF_REGISTER_FAILED) {
        perror("scif_register");
        res.status = RMA_ST_REGISTER;
        res.err = errno;
        send_full(epd, &res, (int)sizeof res);
        return 1;
    }

    struct scif_rma_geom g;
    memset(&g, 0, sizeof g);
    g.rma_off = (int64_t)rma_off;
    g.win_size = (uint64_t)win;
    g.slot_size = (uint32_t)slot;
    g.nslots = (uint32_t)nslots;
    g.ctrl_off = (uint64_t)ctrl_off;
    g.slot_base = (uint64_t)slot_base;
    g.page_size = (uint32_t)ps;
    if (send_full(epd, &g, (int)sizeof g) < 0) return 1;
    printf("scif_rma_sink: window registered at 0x%llx in %.1f ms; "
           "ctrl at +%zu, slots at +%zu; receiving %llu bytes -> %s\n",
           (unsigned long long)rma_off, (t_reg1 - t_reg0) * 1e3, ctrl_off,
           slot_base, (unsigned long long)h.size, path);
    fflush(stdout);

    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror(path);
        res.status = RMA_ST_OPEN;
        res.err = errno;
        send_full(epd, &res, (int)sizeof res);
        return 1;
    }

    volatile uint64_t *const ctrl =
        (volatile uint64_t *)(void *)(winbuf + ctrl_off);
    const uint64_t total_slots = (h.size + slot - 1) / slot;

    /* The checksum runs on its own thread, overlapped with the tmpfs write.
     * Measured on this 7120P: tmpfs writes top out near 370 MB/s (page
     * allocation is globally serialised -- 1..16 threads all give the same
     * number, and posix_fallocate costs the same as writing), while the
     * portable checksum only reaches ~837 MB/s.  Serialised they cap the
     * transfer at ~255 MB/s; overlapped the write is the only limit. */
    struct ck_ctx ctx;
    ctx.win = winbuf;
    ctx.slot_base = slot_base;
    ctx.slot = slot;
    ctx.nslots = nslots;
    ctx.total_slots = total_slots;
    ctx.size = h.size;
    ctx.ready = (volatile int *)calloc((size_t)total_slots, sizeof(int));
    ctx.ckdone = (volatile int *)calloc((size_t)total_slots, sizeof(int));
    ctx.ck = SCIF_RMA_CK_INIT;
    ctx.abort = 0;
    if (!ctx.ready || !ctx.ckdone) {
        perror("calloc");
        res.status = RMA_ST_ALLOC;
        res.err = ENOMEM;
        goto done;
    }
    pthread_t ckth;
    if (pthread_create(&ckth, NULL, ck_worker, &ctx) != 0) {
        perror("pthread_create");
        res.status = RMA_ST_ALLOC;
        res.err = EAGAIN;
        goto done;
    }

    uint64_t remaining = h.size;
    double flush_s = 0.0, ck_wait_s = 0.0, host_wait_s = 0.0;
    const double t_all0 = now_s();
    uint64_t next_report = 1ULL << 30;

    for (uint64_t k = 0; k < total_slots; ++k) {
        /* wait for the host to publish slot k (fence_signal, remote write) */
        double tw = now_s();
        uint64_t it = 0;
        for (;;) {
            if (ctrl[RMA_CTRL_HOST_SEQ / 8] > k) break;
            if (ctrl[RMA_CTRL_ABORT / 8] != 0) {
                res.status = RMA_ST_STALL;
                res.err = ECANCELED;
                goto stop;
            }
            if ((++it & 0xFFFFF) == 0 && now_s() - tw > STALL_TIMEOUT_S) {
                fprintf(stderr, "scif_rma_sink: stalled at slot %llu "
                        "(host_seq=%llu)\n", (unsigned long long)k,
                        (unsigned long long)ctrl[RMA_CTRL_HOST_SEQ / 8]);
                res.status = RMA_ST_STALL;
                res.err = ETIMEDOUT;
                goto stop;
            }
        }
        host_wait_s += now_s() - tw;

        const size_t off = slot_base + (size_t)(k % nslots) * slot;
        const size_t n = (size_t)((remaining < slot) ? remaining : slot);
        const char *src = winbuf + off;

        /* hand the slot to the checksum thread before blocking on the write */
        __sync_synchronize();
        ctx.ready[k] = 1;

        const double t_w0 = now_s();
        size_t done = 0;
        while (done < n) {
            const ssize_t w = write(fd, src + done, n - done);
            if (w <= 0) {
                perror("write");
                res.status = RMA_ST_WRITE;
                res.err = errno;
                goto stop;
            }
            done += (size_t)w;
        }
        flush_s += now_s() - t_w0;

        /* the slot stays read-only until its checksum is in */
        const double t_c0 = now_s();
        while (!ctx.ckdone[k]) { }
        __sync_synchronize();
        ck_wait_s += now_s() - t_c0;

        remaining -= n;
        /* Only now is the slot reusable by the host. */
        ctrl[RMA_CTRL_CARD_SEQ / 8] = k + 1;

        const uint64_t written = h.size - remaining;
        if (written >= next_report) {
            const double el = now_s() - t_all0;
            printf("  %5.1f%%  %.0f MB/s  (flush %.0f MB/s)\n",
                   100.0 * (double)written / (double)h.size,
                   (double)written / 1e6 / el,
                   (double)written / 1e6 / (flush_s + 1e-9));
            fflush(stdout);
            next_report += 1ULL << 30;          /* every GiB */
        }
    }

    if (fsync(fd) != 0 && errno != EINVAL) perror("fsync");
    close(fd);
    pthread_join(ckth, NULL);
    res.status = RMA_ST_OK;
    res.bytes = h.size;
    res.ck = ctx.ck;
    res.slots = total_slots;
    res.flush_us = (uint64_t)(flush_s * 1e6);
    res.total_us = (uint64_t)((now_s() - t_all0) * 1e6);
    res.wait_us = (uint64_t)(host_wait_s * 1e6);
    printf("scif_rma_sink: done. %llu bytes in %.2f s = %.0f MB/s "
           "(flush %.0f MB/s, ck wait %.2f s, host wait %.2f s)\n",
           (unsigned long long)h.size, res.total_us / 1e6,
           (double)h.size / 1e6 / (res.total_us / 1e6),
           (double)h.size / 1e6 / (flush_s + 1e-9), ck_wait_s, host_wait_s);
    fflush(stdout);
    free((void *)ctx.ready);
    free((void *)ctx.ckdone);
    goto done;

stop:
    ctx.abort = 1;
    pthread_join(ckth, NULL);
    free((void *)ctx.ready);
    free((void *)ctx.ckdone);
    res.total_us = (uint64_t)((now_s() - t_all0) * 1e6);
    close(fd);

done:
    if (send_full(epd, &res, (int)sizeof res) < 0)
        fprintf(stderr, "scif_rma_sink: could not report the result\n");
    scif_unregister(epd, rma_off, win);
    scif_close(epd);
    scif_close(lepd);
    free(winbuf);
    return res.status == RMA_ST_OK ? 0 : 1;
}
