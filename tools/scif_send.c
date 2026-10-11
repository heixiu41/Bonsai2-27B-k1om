/* scif_send.c -- Windows host-side file sender over SCIF.
 *
 * Sends one file to tools/scif_sink.c running on the card. Exists because
 * SSH tops out at 9-15 MB/s on the Xeon Phi (no AES-NI, software crypto on
 * 1.2 GHz cores) while SCIF runs at GB/s.
 *
 * The host SCIF library is uSCIF.dll (in System32, installed by the MPSS
 * driver). There is no import .lib shipped, so symbols are resolved with
 * LoadLibrary/GetProcAddress -- no link-time dependency at all.
 *
 * Usage: scif_send.exe <local file> [remote name]
 * Build: zig cc -O2 -o scif_send.exe scif_send.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>

#define SCIF_PORT  20500
#define CHUNK      (16 * 1024 * 1024)
#define SCIF_SEND_BLOCK 1

struct hdr { uint64_t size; char name[248]; };

typedef HANDLE scif_epd_t_w;   /* on Windows hosts, scif_epd_t is a HANDLE */
struct scif_portID { uint16_t node; uint16_t port; };

typedef scif_epd_t_w (WINAPI *scif_open_t)(void);
typedef int (WINAPI *scif_bind_t)(scif_epd_t_w, uint16_t);
typedef int (WINAPI *scif_connect_t)(scif_epd_t_w, struct scif_portID *);
typedef int (WINAPI *scif_send_t)(scif_epd_t_w, const void *, int, int);
typedef int (WINAPI *scif_recv_t)(scif_epd_t_w, void *, int, int);
typedef int (WINAPI *scif_close_t)(scif_epd_t_w);
typedef int (WINAPI *scif_get_nodeIDs_t)(uint16_t *, size_t, uint16_t *);

static scif_open_t         p_scif_open;
static scif_connect_t      p_scif_connect;
static scif_send_t         p_scif_send;
static scif_close_t        p_scif_close;
static scif_get_nodeIDs_t  p_scif_get_nodeIDs;

static double now_s(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: scif_send <local file> [remote name]\n");
        return 2;
    }
    const char *path = argv[1];
    const char *rname = argc > 2 ? argv[2] : NULL;

    HMODULE lib = LoadLibraryA("uSCIF.dll");
    if (!lib) { fprintf(stderr, "uSCIF.dll not found (MPSS driver?)\n"); return 1; }
    p_scif_open        = (scif_open_t)GetProcAddress(lib, "scif_open");
    p_scif_connect     = (scif_connect_t)GetProcAddress(lib, "scif_connect");
    p_scif_send        = (scif_send_t)GetProcAddress(lib, "scif_send");
    p_scif_close       = (scif_close_t)GetProcAddress(lib, "scif_close");
    p_scif_get_nodeIDs = (scif_get_nodeIDs_t)GetProcAddress(lib, "scif_get_nodeIDs");
    if (!p_scif_open || !p_scif_connect || !p_scif_send || !p_scif_close ||
        !p_scif_get_nodeIDs) {
        fprintf(stderr, "uSCIF.dll is missing expected symbols\n");
        return 1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    _fseeki64(f, 0, SEEK_END);
    const uint64_t size = (uint64_t)_ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);

    /* remote name defaults to the basename */
    if (!rname) {
        rname = path;
        for (const char *p = path; *p; ++p)
            if (*p == '\\' || *p == '/') rname = p + 1;
    }

    /* find the card: the online node that is not us */
    uint16_t nodes[16], self = 0;
    const int nnodes = p_scif_get_nodeIDs(nodes, 16, &self);
    if (nnodes < 1) { fprintf(stderr, "no SCIF nodes online\n"); return 1; }
    uint16_t card = 0;
    for (int i = 0; i < nnodes; ++i)
        if (nodes[i] != self) { card = nodes[i]; break; }
    fprintf(stderr, "host node=%d, card node=%d (%d online)\n",
            self, card, nnodes);

    scif_epd_t_w epd = p_scif_open();
    if (!epd || epd == (scif_epd_t_w)-1) {
        fprintf(stderr, "scif_open failed\n"); return 1;
    }
    struct scif_portID dst = { card, SCIF_PORT };
    /* the sink may still be starting: retry for ~60 s */
    int rc = -1;
    for (int i = 0; i < 120 && rc < 0; ++i) {
        rc = p_scif_connect(epd, &dst);
        if (rc < 0) Sleep(500);
    }
    if (rc < 0) {
        fprintf(stderr, "scif_connect failed (is scif_sink running?)\n");
        return 1;
    }
    fprintf(stderr, "connected to card:%d\n", SCIF_PORT);

    struct hdr h;
    h.size = size;
    memset(h.name, 0, sizeof(h.name));
    strncpy(h.name, rname, sizeof(h.name) - 1);
    if (p_scif_send(epd, &h, (int)sizeof(h), SCIF_SEND_BLOCK) != (int)sizeof(h)) {
        fprintf(stderr, "header send failed\n"); return 1;
    }

    void *buf = malloc(CHUNK);
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }
    uint64_t done = 0;
    const double t0 = now_s();
    while (done < size) {
        const int want = (int)((size - done) < CHUNK ? (size - done) : CHUNK);
        if (fread(buf, 1, (size_t)want, f) != (size_t)want) {
            fprintf(stderr, "short read\n"); return 1;
        }
        const int sent = p_scif_send(epd, buf, want, SCIF_SEND_BLOCK);
        if (sent != want) {
            fprintf(stderr, "scif_send: sent %d of %d\n", sent, want);
            return 1;
        }
        done += (uint64_t)want;
        fprintf(stderr, "\r  %5.1f%%  %.0f MB/s", 100.0 * done / size,
                done / 1e6 / (now_s() - t0 + 1e-9));
    }
    const double dt = now_s() - t0;
    fprintf(stderr, "\nsent %llu bytes in %.1f s = %.1f MB/s\n",
            (unsigned long long)size, dt, size / 1e6 / dt);
    fclose(f);
    p_scif_close(epd);
    return 0;
}
