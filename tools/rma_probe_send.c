/* rma_probe_send.c -- Windows host-side feasibility + bandwidth probe for the
 * SCIF RMA direct-transfer path.  Companion to tools/rma_probe_sink.c.
 *
 * Measures, at several RMA chunk sizes:
 *   - plain scif_writeto with SCIF_RMA_SYNC (synchronous, per-chunk completion)
 *   - async scif_writeto + scif_fence_mark/scif_fence_wait
 * then asks the card to hash what it received (integrity) and pokes a magic
 * value into the window through scif_fence_signal(SCIF_SIGNAL_REMOTE).
 *
 * Uses the MPSS Windows SDK header and import library -- the real ABI, not
 * remembered prototypes:
 *   -I "C:\Program Files\Intel\MPSS\sdk\include"   lib_uSCIF.lib
 *
 * Build: zig cc -O2 -I"<sdk>\include" -o rma_probe_send.exe rma_probe_send.c lib_uSCIF.lib
 * Usage: rma_probe_send.exe [win_mb] [chunk_kb,...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* The MPSS SDK scif.h re-typedefs the stdint types unless _STDINT is already
 * defined, and zig cc's <stdint.h> does not define it (the header only takes
 * its "use the compiler's stdint" branch when _MSC_VER >= 1700).  We already
 * have real stdint types at this point, so tell it to stand down. */
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

struct probe_rep {
    int64_t  rma_off;
    uint64_t win;
    uint64_t reg_us;
};

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static int send_full(scif_epd_t e, const void *buf, int n) {
    int sent = 0;
    while (sent < n) {
        int rc = scif_send(e, (char *)buf + sent, n - sent, SCIF_SEND_BLOCK);
        if (rc <= 0) { fprintf(stderr, "scif_send rc=%d errno=%d\n", rc, errno); return -1; }
        sent += rc;
    }
    return sent;
}

static int recv_full(scif_epd_t e, void *buf, int n) {
    int got = 0;
    while (got < n) {
        int rc = scif_recv(e, (char *)buf + got, n - got, SCIF_RECV_BLOCK);
        if (rc <= 0) { fprintf(stderr, "scif_recv rc=%d errno=%d\n", rc, errno); return -1; }
        got += rc;
    }
    return got;
}

static int write_all(scif_epd_t e, int64_t hoff, int64_t roff,
                     uint64_t total, uint64_t chunk, int flags) {
    uint64_t done = 0;
    while (done < total) {
        uint64_t n = total - done;
        if (n > chunk) n = chunk;
        const int rc = scif_writeto(e, hoff + (int64_t)done, (size_t)n,
                                    roff + (int64_t)done, flags);
        if (rc != 0) {
            fprintf(stderr, "  scif_writeto(done=%llu n=%llu) rc=%d errno=%d\n",
                    (unsigned long long)done, (unsigned long long)n, rc, errno);
            return -1;
        }
        done += n;
    }
    return 0;
}

static void report(const char *tag, uint64_t bytes, double dt) {
    printf("  %-34s %8.1f MB/s   (%.3f s for %llu MiB)\n", tag,
           (double)bytes / 1e6 / dt, dt,
           (unsigned long long)(bytes >> 20));
    fflush(stdout);
}

int main(int argc, char **argv) {
    uint64_t win = 512ULL << 20;
    if (argc > 1) win = (uint64_t)strtoull(argv[1], NULL, 0) << 20;

    uint64_t chunks[8];
    int nchunks = 0;
    if (argc > 2) {
        const char *p = argv[2];
        while (p && *p && nchunks < 8) {
            char *q = NULL;
            const unsigned long v = strtoul(p, &q, 0);
            chunks[nchunks++] = (uint64_t)v << 10;
            p = (q && *q == ',') ? q + 1 : NULL;
        }
    } else {
        chunks[nchunks++] = 1ULL << 20;
        chunks[nchunks++] = 4ULL << 20;
        chunks[nchunks++] = 16ULL << 20;
        chunks[nchunks++] = 64ULL << 20;
    }

    printf("rma_probe_send: win=%llu MiB\n", (unsigned long long)(win >> 20));

    /* --- find the card: the online node that is not us ------------------- */
    uint16_t nodes[16], self = 0;
    const int nnodes = scif_get_nodeIDs(nodes, 16, &self);
    if (nnodes < 1) { fprintf(stderr, "no SCIF nodes online\n"); return 1; }
    uint16_t card = 0;
    for (int i = 0; i < nnodes; ++i)
        if (nodes[i] != self) { card = nodes[i]; break; }
    printf("host node=%d card node=%d (%d online)\n", self, card, nnodes);

    scif_epd_t epd = scif_open();
    if (epd == SCIF_OPEN_FAILED) { fprintf(stderr, "scif_open failed\n"); return 1; }
    struct scif_portID dst = { card, SCIF_RMA_PROBE_PORT };
    int rc = -1;
    for (int i = 0; i < 60 && rc < 0; ++i) {
        rc = scif_connect(epd, &dst);
        if (rc < 0) Sleep(500);
    }
    if (rc < 0) { fprintf(stderr, "scif_connect failed (probe sink up?)\n"); return 1; }
    printf("connected to card:%d\n", SCIF_RMA_PROBE_PORT);

    struct probe_rep rep;
    if (recv_full(epd, &rep, (int)sizeof rep) < 0) return 1;
    printf("card registered %llu MiB at rma_off=%lld in %llu us\n",
           (unsigned long long)(rep.win >> 20), (long long)rep.rma_off,
           (unsigned long long)rep.reg_us);
    if (rep.win > win) win = rep.win;

    /* --- host source window: page-aligned, registered once --------------- */
    void *src = VirtualAlloc(NULL, (SIZE_T)win, MEM_COMMIT | MEM_RESERVE,
                             PAGE_READWRITE);
    if (!src) { fprintf(stderr, "VirtualAlloc(%llu) failed\n",
                        (unsigned long long)win); return 1; }
    {
        uint64_t *w = (uint64_t *)src;
        const uint64_t n = win / 8;
        for (uint64_t i = 0; i < n; ++i) w[i] = scif_rma_pattern(i);
    }
    const double t_r0 = now_s();
    const off_t hoff = scif_register(epd, src, (size_t)win, 0,
                                     SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    const double t_r1 = now_s();
    if (hoff == SCIF_REGISTER_FAILED) {
        fprintf(stderr, "host scif_register failed errno=%d\n", errno);
        return 1;
    }
    printf("host registered %llu MiB at loff=%lld in %.3f s\n",
           (unsigned long long)(win >> 20), (long long)hoff, t_r1 - t_r0);

    /* --- 1. synchronous writes at each chunk size ------------------------ */
    printf("[1] scif_writeto + SCIF_RMA_SYNC\n");
    for (int i = 0; i < nchunks; ++i) {
        char tag[64];
        const double t0 = now_s();
        if (write_all(epd, hoff, rep.rma_off, win, chunks[i],
                      SCIF_RMA_SYNC) < 0)
            return 1;
        const double dt = now_s() - t0;
        snprintf(tag, sizeof tag, "sync chunk=%llu KiB",
                 (unsigned long long)(chunks[i] >> 10));
        report(tag, win, dt);
    }

    /* --- 2. asynchronous writes + one fence at the end ------------------- */
    printf("[2] scif_writeto async + scif_fence_mark/wait\n");
    for (int i = 0; i < nchunks; ++i) {
        char tag[64];
        int mark = 0;
        const double t0 = now_s();
        if (write_all(epd, hoff, rep.rma_off, win, chunks[i], 0) < 0) return 1;
        if (scif_fence_mark(epd, SCIF_FENCE_INIT_SELF, &mark) != 0) {
            fprintf(stderr, "  scif_fence_mark rc=%d errno=%d\n", -1, errno);
            return 1;
        }
        if (scif_fence_wait(epd, mark) != 0) {
            fprintf(stderr, "  scif_fence_wait rc=%d errno=%d\n", -1, errno);
            return 1;
        }
        const double dt = now_s() - t0;
        snprintf(tag, sizeof tag, "async chunk=%llu KiB",
                 (unsigned long long)(chunks[i] >> 10));
        report(tag, win, dt);
    }

    /* --- 3. integrity: let the card hash what it actually received ------- */
    printf("[3] card-side verification of %llu MiB\n",
           (unsigned long long)(win >> 20));
    {
        uint64_t cmd = 0, r[4] = { 0, 0, 0, 0 };
        if (send_full(epd, &cmd, 8) < 0) return 1;
        if (recv_full(epd, r, (int)sizeof r) < 0) return 1;
        const uint64_t mine = scif_rma_ck(SCIF_RMA_CK_INIT, src, (size_t)win);
        printf("  host ck=%016llx  card ck=%016llx  %s\n",
               (unsigned long long)mine, (unsigned long long)r[0],
               mine == r[0] ? "MATCH" : "MISMATCH");
        printf("  card saw stale(0xA5) words=%llu  bad words=%llu  "
               "first_bad=%lld\n",
               (unsigned long long)r[1], (unsigned long long)r[2],
               (long long)(int64_t)r[3]);
        fflush(stdout);
    }

    /* --- 4. scif_fence_signal(SCIF_SIGNAL_REMOTE) ------------------------ */
    printf("[4] scif_fence_signal SCIF_SIGNAL_REMOTE\n");
    {
        const uint64_t magic = 0xFEEDFACECAFEBEEFULL;
        const off_t tail = rep.rma_off + (off_t)win - 8;
        uint64_t *w = (uint64_t *)src;
        w[(win / 8) - 1] = magic;
        int rc2 = scif_writeto(epd, hoff + (off_t)(win - 8), 8, tail,
                               SCIF_RMA_SYNC);
        printf("  writeto(8B @tail) rc=%d\n", rc2);
        rc2 = scif_fence_signal(epd, hoff, 0, tail, magic,
                                SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE);
        printf("  fence_signal rc=%d errno=%d\n", rc2, errno);
        if (rc2 == 0) {
            uint64_t cmd = 1, r[4] = { 0, 0, 0, 0 };
            if (send_full(epd, &cmd, 8) < 0) return 1;
            if (recv_full(epd, r, (int)sizeof r) < 0) return 1;
            printf("  card reads tail=0x%016llx  %s\n",
                   (unsigned long long)r[0],
                   r[0] == magic ? "SIGNAL_REMOTE WORKS" : "no signal value");
        }
        fflush(stdout);
    }

    /* --- 5. does a spinning card core observe an RMA-written word? ------- */
    printf("[5] card polls an RMA-written control word\n");
    {
        uint64_t cmd = 3, r[4] = { 0, 0, 0, 0 };
        const off_t ctl = rep.rma_off + (off_t)win - 16;
        if (send_full(epd, &cmd, 8) < 0) return 1;
        Sleep(300);                       /* let the card enter its spin */
        scif_fence_signal(epd, hoff, 0, ctl, 1ULL,
                          SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE);
        Sleep(100);
        scif_fence_signal(epd, hoff, 0, ctl, 2ULL,
                          SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE);
        if (recv_full(epd, r, (int)sizeof r) < 0) return 1;
        printf("  card saw1=%llu saw2=%llu  wait1=%llu us wait2=%llu us  %s\n",
               (unsigned long long)r[0], (unsigned long long)r[1],
               (unsigned long long)r[2], (unsigned long long)r[3],
               (r[0] == 1 && r[1] == 1) ? "POLLING WORKS" : "POLLING FAILED");
        fflush(stdout);
    }

    /* --- 6. card -> host RMA reads --------------------------------------- */
    printf("[6] scif_readfrom (card -> host)\n");
    {
        uint64_t cmd = 4, r[4] = { 0, 0, 0, 0 }, got = 0;
        if (send_full(epd, &cmd, 8) < 0) return 1;
        if (recv_full(epd, r, (int)sizeof r) < 0) return 1;
        const int rc3 = scif_readfrom(epd, hoff + (off_t)(win - 8), 8,
                                      rep.rma_off + (off_t)win - 8,
                                      SCIF_RMA_SYNC);
        memcpy(&got, (char *)src + win - 8, 8);
        printf("  readfrom rc=%d got=0x%016llx expect=0x%016llx  %s\n", rc3,
               (unsigned long long)got, (unsigned long long)r[0],
               (got == r[0]) ? "READFROM WORKS" : "READFROM FAILED");
        fflush(stdout);
    }

    {
        uint64_t cmd = 2;
        send_full(epd, &cmd, 8);
    }
    scif_unregister(epd, hoff, (size_t)win);
    scif_close(epd);
    VirtualFree(src, 0, MEM_RELEASE);
    printf("rma_probe_send: done\n");
    return 0;
}
