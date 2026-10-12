/*
 * gguf.h - minimal GGUF loader for the Bonsai 2 27B k1om port.
 *
 * Scope: exactly what this model needs - the 49 metadata keys, the 851-entry
 * tensor table, an mmap of the whole file, and typed accessors. No writer, no
 * tokenizer-array parsing beyond what the chat template needs.
 *
 * Layout facts, all cross-verified against the official `llama-gguf.exe r`
 * dump of Ternary-Bonsai-2-27B-PQ2_0.gguf:
 *
 *   version    = 3
 *   alignment  = 32
 *   n_kv       = 49
 *   n_tensors  = 851
 *   data offset= 11120992        (= align32(end of tensor-info block))
 *
 *   Tensor `offset` values are RELATIVE to that data offset, NOT absolute file
 *   offsets. Reading them as absolute silently lands inside the wrong tensor
 *   and yields garbage (F32 norms decoding as 1e+25). This was the single most
 *   expensive mistake made while reverse-engineering the format.
 *
 *   GGUF dims are ggml order: dims[0] is the INNERMOST (input) dimension,
 *   the reverse of numpy. `blk.0.attn_qkv.weight` has dims {5120, 10240},
 *   i.e. in=5120, out=10240.
 *
 * k1om notes: the whole file is read into ONE heap buffer (see gguf.c -- it is
 * NOT mmap'd; an earlier version of this comment claimed mmap and was simply
 * wrong). PQ2_0 weights are consumed in place by pq2_0_matvec_knc(); only
 * F32/BF16 activations and small per-layer tables are expanded.
 *
 * THE ALLOCATION IS DELIBERATELY LARGER THAN THE FILE. gguf.c mallocs
 * file_size + 64 and zeroes the tail. Reason: the IMCI PQ2_0 decoder
 * (_mm512_extloadunpacklo/hi_epi32) reads up to 48 bytes past the end of every
 * 34-byte block -- a footprint of [blk, blk+81], independent of alignment,
 * measured on the card against a PROT_NONE guard page. Without the slack, the
 * last block of the last row of the last PQ2_0 tensor would read past the
 * allocation whenever a PQ2_0 tensor happens to be last in the file, which is
 * a runtime SIGSEGV with no build-time warning.
 *
 * So: do NOT "fix" the loader to mmap the file read-only without slack, and do
 * not remove the guard without also fixing the decoder's over-read.
 */
#ifndef BONSAI_GGUF_H
#define BONSAI_GGUF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

/* ---- ggml type ids we handle ---------------------------------------- */
#define GGML_TYPE_F32    0
#define GGML_TYPE_F16    1
#define GGML_TYPE_BF16   30
#define GGML_TYPE_PQ2_0  142
#define GGML_TYPE_PTQ1_0 143

/* our own marker for tensors we can consume, kept separate from ggml ids */
#define BONSAI_DT_UNSUPPORTED 0xFFFFFFFFu

#define GGUF_MAX_DIMS 4
#define GGUF_MAX_NAME 128

typedef struct {
    char     name[GGUF_MAX_NAME];
    int32_t  n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;        /* relative to gguf.data */
    uint64_t nbytes;        /* computed from type + dims */
    uint64_t n_elements;
    /* k1om only: lazily-built per-row PQ2_0 fp16 scale table (the proven
     * "drow precompute" from src/probes/pq2drow.c), dims[1] rows x
     * round_up(dims[0]/128, 16) fp32, built on FIRST mm_run() per tensor
     * (lazy, because at open time the 7.2 GB tmpfs model file still exists
     * and the load peak would OOM the card) and owned/freed by model.c
     * (aligned_f32). NULL on the host build / BONSAI_NO_DROW / before first
     * use; the sentinel PQ2_DROWS_FAILED ((float*)1) marks a permanent
     * allocation failure. Always NULL right after load (calloc). */
    float   *drows;
    /* k1om only: 1 once this tensor's rows were compressed IN PLACE from the
     * 34-byte PQ2_0 block stride to the 32-byte codes-only stride (see
     * pq2_0.h:pq2_repack_rows_inplace).  Set at the same lazy moment as
     * `drows`, because the repack drops the fp16 scale bytes that drows are
     * built from -- drows first, then compress.  Always 0 right after load
     * (calloc), and always 0 on the host build / under BONSAI_NO_DROW. */
    int      packed;
} bonsai_tensor;

typedef struct {
    char     key[96];
    uint32_t type;          /* gguf kv type */

    /* scalars */
    int64_t  i;
    double   f;
    char     s[64];

    /* small int arrays (rope sections, ssm geometry, ...) */
    int64_t  arr[64];
    int32_t  arr_len;
} bonsai_kv;

typedef struct {
    /* header */
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint64_t alignment;
    uint64_t data_offset;       /* absolute file offset of the data region */

    /* metadata */
    bonsai_kv *kv;
    uint64_t  kv_len;

    /* tensor table, sorted by name for binary search */
    bonsai_tensor *tensors;
    uint64_t       tensor_len;

    /* the file itself */
    uint8_t *map;
    uint64_t file_size;

    /* convenience: the geometry we care about */
    int32_t n_layer, n_embd, n_ff, n_vocab;
    int32_t n_head, n_head_kv, head_dim;
    int32_t n_rot;
    int64_t rope_sections[4];
    double  rope_freq_base, rms_eps;
    int32_t ssm_d_conv, ssm_d_state, ssm_d_inner, ssm_dt_rank, ssm_n_group;
    int32_t full_attn_interval;
    int32_t had_block;
    int64_t had_sign_widths[8];
    int32_t had_sign_width_count;

    /* prism.hadamard.sign_values, kept in full. It is only 28672 ints
     * (~115 KB), so the loader caches it rather than skipping it like the
     * 248320-entry tokenizer tables. The widths in had_sign_widths[]
     * partition this array in order. */
    int32_t *had_sign_values;
    int32_t  had_sign_value_count;
} bonsai_gguf;

/* ==================================================================== */
/* little-endian readers (k1om is LE, so these compile to plain loads)   */
/* ==================================================================== */
static inline uint8_t  rd_u8 (const uint8_t *p) { return p[0]; }
static inline uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t rd_u64(const uint8_t *p) {
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}
static inline int32_t  rd_i32(const uint8_t *p) { return (int32_t)rd_u32(p); }
static inline float    rd_f32(const uint8_t *p) {
    union { uint32_t u; float f; } v; v.u = rd_u32(p); return v.f;
}
static inline double   rd_f64(const uint8_t *p) {
    union { uint64_t u; double d; } v; v.u = rd_u64(p); return v.d;
}

/* ggml bf16 -> f32: the value is the top 16 bits of an f32 */
static inline float bonsai_bf16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* ==================================================================== */
/* byte size of a tensor, from its ggml type and dims                    */
/* ==================================================================== */
static inline uint64_t bonsai_tensor_nbytes(uint32_t type, const uint64_t *dims,
                                            int32_t n_dims, uint64_t *n_elem_out) {
    uint64_t n = 1;
    for (int32_t i = 0; i < n_dims; ++i) n *= dims[i];
    if (n_elem_out) *n_elem_out = n;

    switch (type) {
        case GGML_TYPE_F32:  return n * 4;
        case GGML_TYPE_F16:  return n * 2;
        case GGML_TYPE_BF16: return n * 2;
        /* PQ2_0 and PTQ1_0 are both 34 bytes per 128 weights */
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_PTQ1_0:
            if (n % 128) return 0;
            return n / 128 * 34;
        default: return 0;
    }
}

static inline const char *bonsai_type_name(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32:    return "F32";
        case GGML_TYPE_F16:    return "F16";
        case GGML_TYPE_BF16:   return "BF16";
        case GGML_TYPE_PQ2_0:  return "PQ2_0";
        case GGML_TYPE_PTQ1_0: return "PTQ1_0";
        default:               return "?";
    }
}

/* ==================================================================== */
/* metadata value types (GGUF v3)                                        */
/* ==================================================================== */
#define BGGUF_UINT8   0
#define BGGUF_INT8    1
#define BGGUF_UINT16  2
#define BGGUF_INT16   3
#define BGGUF_UINT32  4
#define BGGUF_INT32   5
#define BGGUF_FLOAT32 6
#define BGGUF_BOOL    7
#define BGGUF_STRING  8
#define BGGUF_ARRAY   9
#define BGGUF_UINT64  10
#define BGGUF_INT64   11
#define BGGUF_FLOAT64 12

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int failed;
} bgguf_cur;

static inline void bg_need(bgguf_cur *c, uint64_t n) {
    if (c->failed || (uint64_t)(c->end - c->p) < n) c->failed = 1;
}
static inline uint64_t bg_u64(bgguf_cur *c) {
    bg_need(c, 8); if (c->failed) return 0;
    uint64_t v = rd_u64(c->p); c->p += 8; return v;
}
static inline uint32_t bg_u32(bgguf_cur *c) {
    bg_need(c, 4); if (c->failed) return 0;
    uint32_t v = rd_u32(c->p); c->p += 4; return v;
}
static inline double bg_f64(bgguf_cur *c) {
    bg_need(c, 8); if (c->failed) return 0;
    double v = rd_f64(c->p); c->p += 8; return v;
}
/* returns a pointer into the mapped file; length written to *len */
static inline const char *bg_str(bgguf_cur *c, uint64_t *len) {
    uint64_t n = bg_u64(c);
    if (c->failed || (uint64_t)(c->end - c->p) < n) { c->failed = 1; return NULL; }
    const char *s = (const char *)c->p;
    c->p += n;
    if (len) *len = n;
    return s;
}

/* skip a value of the given type without materialising it (for big arrays) */
static inline void bg_skip(bgguf_cur *c, uint32_t type);

static inline void bg_skip(bgguf_cur *c, uint32_t type) {
    switch (type) {
        case BGGUF_UINT8: case BGGUF_INT8: case BGGUF_BOOL:   bg_need(c,1); c->p += 1; break;
        case BGGUF_UINT16: case BGGUF_INT16:                  bg_need(c,2); c->p += 2; break;
        case BGGUF_UINT32: case BGGUF_INT32: case BGGUF_FLOAT32: bg_need(c,4); c->p += 4; break;
        case BGGUF_UINT64: case BGGUF_INT64: case BGGUF_FLOAT64: bg_need(c,8); c->p += 8; break;
        case BGGUF_STRING: { uint64_t l; bg_str(c, &l); break; }
        case BGGUF_ARRAY: {
            uint32_t et = bg_u32(c);
            uint64_t n  = bg_u64(c);
            if (c->failed) return;
            /* only the scalar element types appear in this model's arrays */
            uint64_t esz = 0;
            switch (et) {
                case BGGUF_UINT8: case BGGUF_INT8: case BGGUF_BOOL:   esz = 1; break;
                case BGGUF_UINT16: case BGGUF_INT16:                  esz = 2; break;
                case BGGUF_UINT32: case BGGUF_INT32: case BGGUF_FLOAT32: esz = 4; break;
                case BGGUF_UINT64: case BGGUF_INT64: case BGGUF_FLOAT64: esz = 8; break;
                case BGGUF_STRING:
                    for (uint64_t i = 0; i < n; ++i) { uint64_t l; bg_str(c, &l); if (c->failed) return; }
                    return;
                default: c->failed = 1; return;
            }
            uint64_t total = esz * n;
            bg_need(c, total);
            c->p += total;
            return;
        }
        default: c->failed = 1; return;
    }
}

/* ==================================================================== */
/* public API (implemented in gguf.c)                                    */
/* ==================================================================== */

/* Load and parse a .gguf, reading the whole file into memory. Returns NULL on
 * failure. The file stays resident for the lifetime of the handle; PQ2_0
 * weights are consumed in place and never expanded. */
bonsai_gguf *bonsai_gguf_load(const char *path);
void         bonsai_gguf_free(bonsai_gguf *g);

/* Binary search by name. Returns NULL when absent. */
const bonsai_tensor *bonsai_tensor_get(const bonsai_gguf *g, const char *name);

/* Pointer to the tensor's first byte inside the loaded file. */
const uint8_t *bonsai_tensor_data(const bonsai_gguf *g, const bonsai_tensor *t);

/* Print the header, geometry and an integrity check (every tensor inside the
 * data region, none spilling past EOF). Returns 0 when all checks pass. */
int bonsai_gguf_selftest(const char *path);

#endif /* BONSAI_GGUF_H */
