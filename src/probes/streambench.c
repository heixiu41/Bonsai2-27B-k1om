/* Emit a fixed volume of bytes on stdout, so the host can measure the
 * throughput of whatever channel is carrying it (SSH vs micnativeloadex/SCIF).
 *
 * Usage: streambench <megabytes>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const long mb = argc > 1 ? atol(argv[1]) : 256;
    const size_t chunk = 1u << 20;                 /* 1 MiB */
    char *buf = (char *)malloc(chunk);
    if (!buf) return 1;
    memset(buf, 0xA5, chunk);
    for (long i = 0; i < mb; ++i) {
        if (fwrite(buf, 1, chunk, stdout) != chunk) return 1;
    }
    fflush(stdout);
    free(buf);
    return 0;
}
