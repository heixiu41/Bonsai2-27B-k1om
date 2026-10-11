/* scif_sink.c -- card-side (k1om) file receiver over SCIF.
 *
 * Companion to tools/scif_send.c (Windows host). Exists because the only
 * other way to get the 7.2 GB model onto the card is SSH at 9-15 MB/s
 * (~7-8 min); SCIF moves data at GB/s (docs/platform/K1OM_TRANSFER_CHANNELS.md).
 *
 * Protocol (one connection, one file):
 *   1. header message: 8-byte LE size + 248-byte NUL-padded file name
 *   2. data: size bytes as a sequence of <=16 MB messages (one scif_send =
 *      one scif_recv; SCIF is message-oriented)
 * The file is written to /tmp/<name>. Prints progress and throughput.
 *
 * Build: icc -mmic -O2 -std=c11 scif_sink.c -o scif_sink -lscif
 * Run:   ./scif_sink            (blocks until one transfer completes)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/types.h>   /* off_t -- scif.h uses it but does not include it */
#include <scif.h>

#define SCIF_PORT  20500
#define CHUNK      (16 * 1024 * 1024)

struct hdr { uint64_t size; char name[248]; };

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* receive exactly n bytes (SCIF preserves message boundaries, so the caller
 * must ask for exactly one message; partial recvs are still handled) */
static int recv_full(scif_epd_t e, void *buf, int n) {
    int got = 0;
    while (got < n) {
        const int rc = scif_recv(e, (char *)buf + got, n - got, SCIF_RECV_BLOCK);
        if (rc <= 0) { perror("scif_recv"); return -1; }
        got += rc;
    }
    return got;
}

int main(void) {
    scif_epd_t lepd = scif_open();
    if (lepd == SCIF_OPEN_FAILED) { perror("scif_open"); return 1; }
    const int port = scif_bind(lepd, SCIF_PORT);
    if (port < 0) { perror("scif_bind"); return 1; }
    if (scif_listen(lepd, 1) < 0) { perror("scif_listen"); return 1; }
    printf("scif_sink: listening on port %d\n", port);
    fflush(stdout);

    struct scif_portID peer;
    scif_epd_t epd;
    if (scif_accept(lepd, &peer, &epd, SCIF_ACCEPT_SYNC) < 0) {
        perror("scif_accept"); return 1;
    }
    printf("scif_sink: connection from node %d\n", peer.node);
    fflush(stdout);

    struct hdr h;
    if (recv_full(epd, &h, (int)sizeof(h)) < 0) return 1;
    h.name[sizeof(h.name) - 1] = '\0';
    char path[300];
    snprintf(path, sizeof(path), "/tmp/%s", h.name);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    printf("scif_sink: receiving %llu bytes -> %s\n",
           (unsigned long long)h.size, path);
    fflush(stdout);

    void *buf = malloc(CHUNK);
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }

    uint64_t done = 0, next_report = 0;
    const double t0 = now_s();
    while (done < h.size) {
        const int want = (int)((h.size - done) < CHUNK ? (h.size - done) : CHUNK);
        if (recv_full(epd, buf, want) < 0) { fclose(f); return 1; }
        if (fwrite(buf, 1, (size_t)want, f) != (size_t)want) {
            perror("fwrite"); fclose(f); return 1;
        }
        done += (uint64_t)want;
        if (done >= next_report) {
            printf("  %5.1f%%  %.0f MB/s\n", 100.0 * done / h.size,
                   done / 1e6 / (now_s() - t0 + 1e-9));
            fflush(stdout);
            next_report = done + (uint64_t)1024 * 1024 * 1024;
        }
    }
    fclose(f);
    const double dt = now_s() - t0;
    printf("scif_sink: done. %llu bytes in %.1f s = %.1f MB/s\n",
           (unsigned long long)h.size, dt, h.size / 1e6 / dt);
    scif_close(epd);
    scif_close(lepd);
    return 0;
}
