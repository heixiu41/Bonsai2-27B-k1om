/* scif_rma.h -- shared constants and helpers for the SCIF RMA direct-transfer
 * path (card-side sink + Windows host sender).
 *
 * Included by BOTH sides, so it must stay portable: plain C99/C11, no platform
 * headers, no allocation.  Types/constants follow the MPSS 3.8.6 headers --
 *   card : C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\include\scif.h
 *   host : C:\Program Files\Intel\MPSS\sdk\include\scif.h
 * (constraint 3: no remembered values).
 */
#ifndef SCIF_RMA_H
#define SCIF_RMA_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- ports -------------------------------------------------------------- */
#define SCIF_RMA_PORT       20500   /* live path: same port as the legacy sink */
#define SCIF_RMA_PROBE_PORT 20501   /* probe only                              */

/* ---- handshake (constraint 2: byte-identical to the legacy message sink) - */
struct scif_hdr {
    uint64_t size;        /* file size in bytes */
    char     name[248];   /* NUL-padded remote basename */
};

/* ---- geometry the card returns after it registered its ring window -----
 * Window layout (all offsets relative to the start of the registration):
 *
 *   ctrl_off  .. ctrl_off + RMA_CTRL_BYTES    counters (never overlapped)
 *   slot_base .. slot_base + nslots*slot_size slot ring
 *
 * The control block MUST sit outside the slot range: both sides DMA into this
 * one window, and an overlap silently turns file bytes into a bogus credit
 * counter (which is exactly how the first cut of this broke).
 */
struct scif_rma_geom {
    int64_t  rma_off;     /* window offset in the card's registered space   */
    uint64_t win_size;    /* bytes registered (control area + all slots)    */
    uint64_t ctrl_off;    /* window-relative offset of the control area     */
    uint64_t slot_base;   /* window-relative offset of slot 0               */
    uint32_t slot_size;   /* bytes per slot (page multiple)                 */
    uint32_t nslots;      /* ring depth                                     */
    uint32_t page_size;   /* card page size used for the alignment checks   */
    uint32_t _pad;
};

/* Control area layout: little-endian uint64 slots at ctrl_off + k*8.
 * It lives inside the card's registered window, so the host can touch it with
 * scif_fence_signal(SCIF_SIGNAL_REMOTE) / scif_readfrom instead of scif_send. */
#define RMA_CTRL_HOST_SEQ 0    /* card -> host: slots the host has written  */
#define RMA_CTRL_CARD_SEQ 8    /* host -> card: slots the card has consumed */
#define RMA_CTRL_ABORT    16   /* host -> card: non-zero => stop flushing   */
#define RMA_CTRL_BYTES    64

/* ---- final report the card sends once every byte has been flushed ------ */
#define RMA_ST_OK        0
#define RMA_ST_ARGS     -1    /* bad header / geometry                       */
#define RMA_ST_ALLOC    -2    /* could not allocate the window               */
#define RMA_ST_REGISTER -3    /* scif_register failed                        */
#define RMA_ST_OPEN     -4    /* could not create the destination file       */
#define RMA_ST_WRITE    -5    /* short write (tmpfs full?)                   */
#define RMA_ST_STALL    -6    /* host stopped publishing slots (timeout)     */

struct rma_result {
    int64_t  status;      /* RMA_ST_*                                        */
    uint64_t bytes;       /* bytes written to the file                       */
    uint64_t ck;          /* checksum over exactly those bytes               */
    uint64_t slots;       /* slots flushed                                   */
    uint64_t flush_us;    /* time spent copying window -> file               */
    uint64_t total_us;    /* whole data phase                                */
    int64_t  err;         /* errno at the point of failure                   */
    uint64_t wait_us;     /* time the card spent waiting for the host        */
};

/* ---- integrity ---------------------------------------------------------
 * Rolling 64-bit checksum, deliberately multiply-free: the KNC cores run at
 * 1.2 GHz with slow 64-bit integer multiply, and this runs over the whole
 * 7.2 GB on both sides.  Detects inadvertent corruption, not tampering --
 * the card's md5sum (BusyBox ships no sha256sum) is the cryptographic check.
 */
#define SCIF_RMA_CK_INIT 0xCBF29CE484222325ULL

static inline uint64_t scif_rma_ck_mix(uint64_t h, uint64_t v) {
    h ^= v;
    h += 0x9E3779B97F4A7C15ULL;
    h = (h << 31) | (h >> 33);
    return h;
}

static inline uint64_t scif_rma_ck(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, b + i, 8);
        h = scif_rma_ck_mix(h, v);
    }
    for (; i < n; ++i) {
        h = scif_rma_ck_mix(h, (uint64_t)b[i]);
    }
    return h;
}

/* Deterministic test pattern used by the probe (not by the live path). */
static inline uint64_t scif_rma_pattern(uint64_t i) {
    uint64_t v = i * 0x9E3779B97F4A7C15ULL;
    v ^= v >> 29;
    v *= 0xBF58476D1CE4E5B9ULL;
    v ^= v >> 32;
    return v;
}

#endif /* SCIF_RMA_H */
