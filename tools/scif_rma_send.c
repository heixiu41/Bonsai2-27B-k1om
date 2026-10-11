/* scif_rma_send.c -- Windows host-side file sender over SCIF RMA.
 *
 * Replacement for tools/scif_send.c.  Same command line and same handshake
 * (8-byte LE size + 248-byte name on port 20500), but the file no longer
 * travels as 16 MB scif_send messages: the host registers its source buffers
 * once and DMAs whole ring slots into the card's registered window with
 * scif_writeto, then publishes progress with
 * scif_fence_signal(SCIF_SIGNAL_REMOTE) -- so the control word cannot land
 * before the slot data it refers to.
 *
 * Flow control comes back the other way through the same window:
 * ctrl[RMA_CTRL_CARD_SEQ] is read with scif_readfrom, and the host never
 * writes into a slot the card has not finished flushing.
 *
 * Built against the real MPSS Windows SDK header and import library:
 *   -I "C:\Program Files\Intel\MPSS\sdk\include"   lib_uSCIF.lib
 *
 * Usage: scif_rma_send.exe <local file> [remote name]
 * Build: zig cc -O2 -I"<sdk>\include" -o scif_rma_send.exe scif_rma_send.c lib_uSCIF.lib
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* The MPSS SDK scif.h re-typedefs the stdint types unless _STDINT is already
 * defined, and zig cc's <stdint.h> does not define it (the header only takes
 * its "use the compiler's stdint" branch when _MSC_VER >= 1700). */
#ifndef _STDINT
#define _STDINT
#define SCIF_RMA_UNDEF_STDINT
#endif
#include <windows.h>
#include <scif.h>
#ifdef SCIF_RMA_UNDEF_STDINT
#undef _STDINT
#undef SCIF_RMA_UNDEF_STDINT
#endif

#include "scif_rma.h"

#define NBUF       4          /* registered source buffers kept in flight  */
#define CONNECT_TRIES 120     /* the sink may still be starting           */
#define STALL_TIMEOUT_S 300.0

typedef char assert_hdr_size[(sizeof(struct scif_hdr) == 256) ? 1 : -1];
typedef char assert_geom_size[(sizeof(struct scif_rma_geom) == 48) ? 1 : -1];
typedef char assert_res_size[(sizeof(struct rma_result) == 64) ? 1 : -1];

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static int send_full(scif_epd_t e, const void *buf, int n) {
    int sent = 0;
    while (sent < n) {
        const int rc = scif_send(e, (char *)buf + sent, n - sent, SCIF_SEND_BLOCK);
        if (rc <= 0) { fprintf(stderr, "scif_send rc=%d errno=%d\n", rc, errno); return -1; }
        sent += rc;
    }
    return sent;
}

static int recv_full(scif_epd_t e, void *buf, int n) {
    int got = 0;
    while (got < n) {
        const int rc = scif_recv(e, (char *)buf + got, n - got, SCIF_RECV_BLOCK);
        if (rc <= 0) { fprintf(stderr, "scif_recv rc=%d errno=%d\n", rc, errno); return -1; }
        got += rc;
    }
    return got;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: scif_rma_send <local file> [remote name]\n");
        return 2;
    }
    const char *path = argv[1];
    const char *rname = argc > 2 ? argv[2] : NULL;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    _fseeki64(f, 0, SEEK_END);
    const uint64_t size = (uint64_t)_ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);

    if (!rname) {
        rname = path;
        for (const char *p = path; *p; ++p)
            if (*p == '\\' || *p == '/') rname = p + 1;
    }

    uint16_t nodes[16], self = 0;
    const int nnodes = scif_get_nodeIDs(nodes, 16, &self);
    if (nnodes < 1) { fprintf(stderr, "no SCIF nodes online\n"); return 1; }
    uint16_t card = 0;
    for (int i = 0; i < nnodes; ++i)
        if (nodes[i] != self) { card = nodes[i]; break; }
    fprintf(stderr, "host node=%d, card node=%d (%d online)\n", self, card, nnodes);

    scif_epd_t epd = scif_open();
    if (epd == SCIF_OPEN_FAILED) { fprintf(stderr, "scif_open failed\n"); return 1; }
    struct scif_portID dst = { card, SCIF_RMA_PORT };
    int rc = -1;
    for (int i = 0; i < CONNECT_TRIES && rc < 0; ++i) {
        rc = scif_connect(epd, &dst);
        if (rc < 0) Sleep(500);
    }
    if (rc < 0) {
        fprintf(stderr, "scif_connect failed (is scif_rma_sink running?)\n");
        return 1;
    }

    struct scif_hdr h;
    memset(&h, 0, sizeof h);
    h.size = size;
    strncpy(h.name, rname, sizeof(h.name) - 1);
    if (send_full(epd, &h, (int)sizeof h) != (int)sizeof h) return 1;

    struct scif_rma_geom g;
    if (recv_full(epd, &g, (int)sizeof g) != (int)sizeof g) return 1;
    if (g.rma_off <= 0 || g.slot_size == 0 || g.nslots < 2) {
        /* the card reported a failure instead of a geometry */
        const struct rma_result *bad = (const struct rma_result *)&g;
        fprintf(stderr, "card refused the transfer: status=%lld err=%lld\n",
                (long long)bad->status, (long long)bad->err);
        return 1;
    }
    /* Refuse a window whose control block overlaps the slot ring: that layout
     * makes file bytes masquerade as the credit counter. */
    if (g.ctrl_off + RMA_CTRL_BYTES > g.slot_base ||
        g.slot_base + (uint64_t)g.nslots * g.slot_size > g.win_size) {
        fprintf(stderr, "card geometry is inconsistent: ctrl=%llu slot_base=%llu "
                "nslots=%u slot=%u win=%llu\n",
                (unsigned long long)g.ctrl_off,
                (unsigned long long)g.slot_base, g.nslots, g.slot_size,
                (unsigned long long)g.win_size);
        return 1;
    }
    fprintf(stderr, "card window: rma_off=0x%llx  %llu slots x %u KiB "
            "(ctrl +%llu, slots +%llu, page %u)\n",
            (unsigned long long)g.rma_off, (unsigned long long)g.nslots,
            g.slot_size >> 10, (unsigned long long)g.ctrl_off,
            (unsigned long long)g.slot_base, g.page_size);

    const size_t slot = g.slot_size;
    const size_t page = g.page_size ? g.page_size : 4096;
    const size_t scratch = (size_t)NBUF * slot;      /* tail page, read target */
    const size_t host_win = scratch + page;
    const uint64_t total_slots = (size + slot - 1) / slot;

    void *src = VirtualAlloc(NULL, host_win, MEM_COMMIT | MEM_RESERVE,
                             PAGE_READWRITE);
    if (!src) { fprintf(stderr, "VirtualAlloc(%zu) failed\n", host_win); return 1; }
    char *p_scratch = (char *)src + scratch;

    const off_t hoff = scif_register(epd, src, host_win, 0,
                                     SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    if (hoff == SCIF_REGISTER_FAILED) {
        fprintf(stderr, "host scif_register failed errno=%d\n", errno);
        return 1;
    }

    int marks[NBUF];
    int have_mark[NBUF];
    memset(have_mark, 0, sizeof have_mark);

    fprintf(stderr, "sending %llu bytes in %llu slots of %u KiB ...\n",
            (unsigned long long)size, (unsigned long long)total_slots,
            g.slot_size >> 10);

    uint64_t remaining = size, ck = SCIF_RMA_CK_INIT;
    double write_s = 0.0, read_s = 0.0, wait_s = 0.0;
    const double t0 = now_s();
    uint64_t next_report = 1ULL << 30;
    int failed = 0;

    for (uint64_t k = 0; k < total_slots; ++k) {
        const int bi = (int)(k % NBUF);

        if (k >= g.nslots) {
            /* the previous occupant of this slot must be on the card's file */
            const uint64_t need = k - g.nslots + 1;
            const double tw = now_s();
            for (;;) {
                uint64_t have = 0;
                if (scif_readfrom(epd, hoff + (off_t)scratch, 8,
                                  g.rma_off + (off_t)(g.ctrl_off + RMA_CTRL_CARD_SEQ),
                                  SCIF_RMA_SYNC) != 0) {
                    fprintf(stderr, "scif_readfrom(card_seq) failed errno=%d\n", errno);
                    failed = 1;
                    break;
                }
                memcpy(&have, p_scratch, 8);
                if (have > total_slots) {
                    /* The credit counter must be a slot count.  Anything else
                     * means the two sides disagree about the window layout and
                     * the host is about to overwrite a slot the card still
                     * needs -- fail loudly instead of racing. */
                    fprintf(stderr, "implausible card_seq=%llu (> %llu slots): "
                            "window layout mismatch\n",
                            (unsigned long long)have,
                            (unsigned long long)total_slots);
                    failed = 1;
                    break;
                }
                if (have >= need) break;
                if (now_s() - tw > STALL_TIMEOUT_S) {
                    fprintf(stderr, "card stopped consuming at slot %llu "
                            "(card_seq=%llu need=%llu)\n",
                            (unsigned long long)k, (unsigned long long)have,
                            (unsigned long long)need);
                    failed = 1;
                    break;
                }
            }
            wait_s += now_s() - tw;
            if (failed) break;
        }
        if (k >= NBUF && have_mark[bi]) {
            if (scif_fence_wait(epd, marks[bi]) != 0) {
                fprintf(stderr, "scif_fence_wait failed errno=%d\n", errno);
                failed = 1;
                break;
            }
            have_mark[bi] = 0;
        }

        const size_t n = (size_t)((remaining < slot) ? remaining : slot);
        char *buf = (char *)src + (size_t)bi * slot;

        const double tr0 = now_s();
        if (fread(buf, 1, n, f) != n) {
            fprintf(stderr, "short read at slot %llu\n", (unsigned long long)k);
            failed = 1;
            break;
        }
        read_s += now_s() - tr0;
        ck = scif_rma_ck(ck, buf, n);

        const double tw0 = now_s();
        /* async: the DMA runs while the next slot is read from disk */
        if (scif_writeto(epd, hoff + (off_t)((size_t)bi * slot), n,
                         g.rma_off + (off_t)(g.slot_base +
                                             (size_t)(k % g.nslots) * slot),
                         0) != 0) {
            fprintf(stderr, "scif_writeto(slot %llu) failed errno=%d\n",
                    (unsigned long long)k, errno);
            failed = 1;
            break;
        }
        if (scif_fence_mark(epd, SCIF_FENCE_INIT_SELF, &marks[bi]) != 0) {
            fprintf(stderr, "scif_fence_mark failed errno=%d\n", errno);
            failed = 1;
            break;
        }
        have_mark[bi] = 1;
        write_s += now_s() - tw0;

        /* publish "slot k is complete" to the card; the fence guarantees the
         * value cannot be written before slot k's data has landed */
        if (scif_fence_signal(epd, hoff, 0,
                              g.rma_off + (off_t)(g.ctrl_off + RMA_CTRL_HOST_SEQ),
                              k + 1,
                              SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE) != 0) {
            fprintf(stderr, "scif_fence_signal(slot %llu) failed errno=%d\n",
                    (unsigned long long)k, errno);
            failed = 1;
            break;
        }

        remaining -= n;
        const uint64_t done = size - remaining;
        if (done >= next_report) {
            const double el = now_s() - t0;
            fprintf(stderr, "\r  %5.1f%%  %.0f MB/s   ",
                    100.0 * (double)done / (double)size,
                    (double)done / 1e6 / el);
            next_report += 1ULL << 30;
        }
    }
    fprintf(stderr, "\n");

    struct rma_result res;
    memset(&res, 0, sizeof res);
    if (!failed) {
        if (recv_full(epd, &res, (int)sizeof res) != (int)sizeof res) {
            fprintf(stderr, "card did not report a result\n");
            failed = 1;
        }
    }

    const double dt = now_s() - t0;
    int exit_code = 1;
    if (!failed && res.status == RMA_ST_OK) {
        const int bytes_ok = (res.bytes == size);
        const int ck_ok = (res.ck == ck);
        fprintf(stderr,
                "sent %llu bytes in %.1f s = %.1f MB/s (%.1f MiB/s)\n"
                "  card: %llu slots, flush %.1f s (%.0f MB/s), total %.1f s, "
                "card-wait-host %.2f s\n"
                "  integrity: bytes %s, checksum %s (host %016llx card %016llx)\n"
                "  host time: read %.1f s, rma-issue %.1f s, ring-wait %.1f s\n",
                (unsigned long long)size, dt, (double)size / 1e6 / dt,
                (double)size / (1 << 20) / dt,
                (unsigned long long)res.slots, res.flush_us / 1e6,
                (double)size / 1e6 / (res.flush_us / 1e6 + 1e-9),
                res.total_us / 1e6,
                res.wait_us / 1e6,
                bytes_ok ? "OK" : "MISMATCH", ck_ok ? "OK" : "MISMATCH",
                (unsigned long long)ck, (unsigned long long)res.ck,
                read_s, write_s, wait_s);
        exit_code = (bytes_ok && ck_ok) ? 0 : 1;
    } else if (!failed) {
        fprintf(stderr, "card reported status=%lld err=%lld after %llu bytes\n",
                (long long)res.status, (long long)res.err,
                (unsigned long long)res.bytes);
    }

    scif_unregister(epd, hoff, host_win);
    scif_close(epd);
    VirtualFree(src, 0, MEM_RELEASE);
    fclose(f);
    return exit_code;
}
