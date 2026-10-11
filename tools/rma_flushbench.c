/* rma_flushbench.c -- card-side (k1om) benchmark for the RMA sink's flush path.
 *
 * The 256 MB RMA window is not the bottleneck; the card copying each slot into
 * tmpfs is.  This measures, on the real hardware:
 *   - single-threaded sequential write() at several chunk sizes
 *   - multi-threaded pwrite() to disjoint offsets (what a threaded flusher
 *     would do) at 2/4/8 threads
 *   - read-back bandwidth
 *   - the cost of the transfer's rolling checksum over one slot
 * so the sink's design follows numbers instead of a guess.
 *
 * Build: icc -mmic -O2 -std=c11 rma_flushbench.c -o rma_flushbench -lpthread
 * Run:   ./rma_flushbench [total_mb] [chunk_mb]
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
#include <pthread.h>

#include "scif_rma.h"

#define PATH "/tmp/flushbench.bin"

static char  *g_src;
static int    g_fd = -1;
static size_t g_total, g_chunk;
static int    g_nthreads;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Each thread owns a disjoint byte range and pwrites it at the matching file
 * offset -- exactly the shape a threaded sink flusher would have. */
static void *worker(void *arg) {
    const long id = (long)arg;
    const size_t region = (g_total + (size_t)g_nthreads - 1) / (size_t)g_nthreads;
    size_t off = (size_t)id * region;
    if (off >= g_total) return NULL;
    const size_t end = (off + region < g_total) ? off + region : g_total;

    while (off < end) {
        const size_t n = (end - off < g_chunk) ? (end - off) : g_chunk;
        size_t done = 0;
        while (done < n) {
            const ssize_t w = pwrite(g_fd, g_src + done, n - done,
                                     (off_t)(off + done));
            if (w <= 0) { perror("pwrite"); return NULL; }
            done += (size_t)w;
        }
        off += n;
    }
    return NULL;
}

static void bench_pwrite(int nt) {
    pthread_t th[16];
    if (nt > 16) nt = 16;
    g_nthreads = nt;
    g_fd = open(PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_fd < 0) { perror("open"); return; }
    const double t0 = now_s();
    for (long i = 0; i < nt; ++i) pthread_create(&th[i], NULL, worker, (void *)i);
    for (int i = 0; i < nt; ++i) pthread_join(th[i], NULL);
    const double dt = now_s() - t0;
    close(g_fd);
    printf("  pwrite %d threads              %8.0f MB/s\n", nt,
           (double)g_total / 1e6 / dt);
    fflush(stdout);
}

int main(int argc, char **argv) {
    g_total = 512ULL << 20;
    g_chunk = 32ULL << 20;
    if (argc > 1) g_total = (size_t)strtoull(argv[1], NULL, 0) << 20;
    if (argc > 2) g_chunk = (size_t)strtoull(argv[2], NULL, 0) << 20;
    if (g_chunk > g_total) g_chunk = g_total;

    g_src = (char *)malloc(g_chunk);
    if (!g_src) { perror("malloc"); return 1; }
    memset(g_src, 0x5A, g_chunk);

    printf("rma_flushbench: total=%zu MiB chunk=%zu MiB\n",
           g_total >> 20, g_chunk >> 20);
    fflush(stdout);

    {
        const double t0 = now_s();
        uint64_t ck = SCIF_RMA_CK_INIT;
        for (int i = 0; i < 8; ++i) ck = scif_rma_ck(ck, g_src, g_chunk);
        const double dt = now_s() - t0;
        printf("  checksum (portable)            %8.0f MB/s  (%.1f ms per %zu MiB)\n",
               (double)g_chunk * 8 / 1e6 / dt, dt / 8 * 1e3, g_chunk >> 20);
        fflush(stdout);
    }

    {
        static const size_t sizes[] = { 1ULL << 20, 4ULL << 20, 16ULL << 20,
                                        32ULL << 20, 64ULL << 20 };
        for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
            if (sizes[s] > g_chunk) continue;   /* source buffer is g_chunk big */
            const size_t c = sizes[s] > g_total ? g_total : sizes[s];
            const int fd = open(PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) { perror("open"); return 1; }
            const double t0 = now_s();
            size_t done = 0;
            while (done < g_total) {
                const size_t n = (g_total - done < c) ? (g_total - done) : c;
                size_t w = 0;
                while (w < n) {
                    const ssize_t r = write(fd, g_src + w, n - w);
                    if (r <= 0) { perror("write"); return 1; }
                    w += (size_t)r;
                }
                done += n;
            }
            const double dt = now_s() - t0;
            close(fd);
            printf("  write  1 thread, %3zu MiB      %8.0f MB/s\n", c >> 20,
                   (double)g_total / 1e6 / dt);
            fflush(stdout);
        }
    }

    /* Is the wall page allocation or the copy?  Rewrite the same file: every
     * page already exists, so nothing has to be allocated on this pass. */
    {
        const int fd = open(PATH, O_WRONLY);   /* no O_TRUNC */
        if (fd >= 0) {
            const double t0 = now_s();
            size_t done = 0;
            while (done < g_total) {
                const size_t n = (g_total - done < g_chunk) ? (g_total - done) : g_chunk;
                size_t w = 0;
                while (w < n) {
                    const ssize_t r = write(fd, g_src + w, n - w);
                    if (r <= 0) { perror("write"); break; }
                    w += (size_t)r;
                }
                done += n;
            }
            const double dt = now_s() - t0;
            close(fd);
            printf("  rewrite same file (pages hot)  %8.0f MB/s\n",
                   (double)g_total / 1e6 / dt);
            fflush(stdout);
        }
    }

    /* Plain RAM copy, as the ceiling the tmpfs path could in principle reach. */
    {
        char *dst = (char *)malloc(g_chunk);
        if (!dst) { perror("malloc"); return 1; }
        const double t0 = now_s();
        size_t done = 0;
        while (done < g_total) {
            memcpy(dst, g_src, g_chunk);
            done += g_chunk;
        }
        const double dt = now_s() - t0;
        free(dst);
        printf("  memcpy in RAM                  %8.0f MB/s\n",
               (double)g_total / 1e6 / dt);
        fflush(stdout);
    }

    /* Does pre-allocating the pages (no data copy) pay for itself? */
    {
        unlink(PATH);
        const int fd = open(PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { perror("open"); }
        else {
            const double t0 = now_s();
            const int r = posix_fallocate(fd, 0, (off_t)g_total);
            const double dt = now_s() - t0;
            if (r != 0)
                printf("  posix_fallocate FAILED: %s\n", strerror(r));
            else {
                printf("  posix_fallocate                %8.0f MB/s\n",
                       (double)g_total / 1e6 / dt);
                const double t1 = now_s();
                size_t done = 0;
                while (done < g_total) {
                    const size_t n =
                        (g_total - done < g_chunk) ? (g_total - done) : g_chunk;
                    size_t w = 0;
                    while (w < n) {
                        const ssize_t rr = write(fd, g_src + w, n - w);
                        if (rr <= 0) { perror("write"); break; }
                        w += (size_t)rr;
                    }
                    done += n;
                }
                printf("  write after fallocate          %8.0f MB/s\n",
                       (double)g_total / 1e6 / (now_s() - t1));
            }
            close(fd);
        }
        fflush(stdout);
    }

    bench_pwrite(1);
    bench_pwrite(2);
    bench_pwrite(4);
    bench_pwrite(8);
    bench_pwrite(16);

    {
        const int fd = open(PATH, O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        const double t0 = now_s();
        size_t done = 0;
        while (done < g_total) {
            const size_t n = (g_total - done < g_chunk) ? (g_total - done) : g_chunk;
            if (read(fd, g_src, n) != (ssize_t)n) { perror("read"); return 1; }
            done += n;
        }
        const double dt = now_s() - t0;
        close(fd);
        printf("  read   1 thread                %8.0f MB/s\n",
               (double)g_total / 1e6 / dt);
    }

    unlink(PATH);
    free(g_src);
    return 0;
}
