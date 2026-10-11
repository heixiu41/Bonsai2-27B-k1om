/*
 * gguf.c - GGUF loader implementation for the Bonsai 2 27B k1om port.
 *
 * Design note on memory: the 7120P has 16 GB of GDDR5 and the model is 7.2 GB,
 * so the weights are read straight into a heap buffer rather than mmap'd. That
 * keeps residency explicit and avoids relying on page-cache behaviour for a
 * file that lives on the card's (RAM-backed) filesystem. PQ2_0 weights are
 * never expanded, so the resident cost stays at the on-disk 7.2 GB.
 */
#include "gguf.h"

#include <errno.h>

/* ------------------------------------------------------------------ */
/* metadata helpers                                                    */
/* ------------------------------------------------------------------ */
static const bonsai_kv *kv_find(const bonsai_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->kv_len; ++i) {
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    }
    return NULL;
}

static int64_t kv_int(const bonsai_gguf *g, const char *key, int64_t dflt) {
    const bonsai_kv *k = kv_find(g, key);
    if (!k) return dflt;
    if (k->type == BGGUF_FLOAT32 || k->type == BGGUF_FLOAT64) return (int64_t)k->f;
    return k->i;
}

static double kv_float(const bonsai_gguf *g, const char *key, double dflt) {
    const bonsai_kv *k = kv_find(g, key);
    if (!k) return dflt;
    if (k->type == BGGUF_FLOAT32 || k->type == BGGUF_FLOAT64) return k->f;
    return (double)k->i;
}

static int tensor_cmp(const void *a, const void *b) {
    return strcmp(((const bonsai_tensor *)a)->name,
                  ((const bonsai_tensor *)b)->name);
}

/* ------------------------------------------------------------------ */
/* header / metadata / tensor table                                    */
/* ------------------------------------------------------------------ */
static int gguf_parse(bonsai_gguf *g) {
    bgguf_cur c = { g->map, g->map + g->file_size, 0 };

    if (g->file_size < 24 || memcmp(c.p, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf: not a GGUF file\n");
        return -1;
    }
    c.p += 4;
    g->version   = bg_u32(&c);
    g->n_tensors = bg_u64(&c);
    g->n_kv      = bg_u64(&c);
    if (c.failed) { fprintf(stderr, "gguf: header truncated\n"); return -1; }
    if (g->version != 3) {
        fprintf(stderr, "gguf: unsupported version %u\n", g->version);
        return -1;
    }
    fprintf(stderr, "gguf: v%u n_tensors=%llu n_kv=%llu\n", g->version,
            (unsigned long long)g->n_tensors, (unsigned long long)g->n_kv);

    /* ---- metadata ---- */
    g->kv = (bonsai_kv *)calloc(g->n_kv ? g->n_kv : 1, sizeof(bonsai_kv));
    if (!g->kv) { fprintf(stderr, "gguf: kv alloc failed\n"); return -1; }

    uint64_t wi = 0;
    for (uint64_t i = 0; i < g->n_kv; ++i) {
        uint64_t klen = 0;
        const char *ks = bg_str(&c, &klen);
        if (c.failed) { fprintf(stderr, "gguf: kv[%llu] key truncated\n",
                                (unsigned long long)i); return -1; }
        uint32_t vt = bg_u32(&c);
        if (c.failed) { fprintf(stderr, "gguf: kv[%llu] type truncated\n",
                                (unsigned long long)i); return -1; }

        bonsai_kv *k = &g->kv[wi];
        size_t copy = klen < sizeof(k->key) - 1 ? klen : sizeof(k->key) - 1;
        memcpy(k->key, ks, copy);
        k->key[copy] = '\0';
        k->type = vt;
        k->arr_len = 0;

        /* scalar values must be bounds-checked BEFORE they are read: a file that
         * ends at the value field used to be read past the end of the mapping. */
        switch (vt) {
            case BGGUF_UINT8: case BGGUF_INT8: case BGGUF_BOOL:
                bg_need(&c, 1);
                if (c.failed) { fprintf(stderr, "gguf: kv[%llu] '%s' value truncated\n",
                                        (unsigned long long)i, k->key); return -1; }
                k->i = (int64_t)c.p[0]; c.p += 1; break;
            case BGGUF_UINT16: case BGGUF_INT16:
                bg_need(&c, 2);
                if (c.failed) { fprintf(stderr, "gguf: kv[%llu] '%s' value truncated\n",
                                        (unsigned long long)i, k->key); return -1; }
                k->i = (int64_t)rd_u16(c.p); c.p += 2; break;
            case BGGUF_UINT32: case BGGUF_INT32:
                bg_need(&c, 4);
                if (c.failed) { fprintf(stderr, "gguf: kv[%llu] '%s' value truncated\n",
                                        (unsigned long long)i, k->key); return -1; }
                k->i = (int64_t)rd_u32(c.p); c.p += 4; break;
            case BGGUF_FLOAT32:
                bg_need(&c, 4);
                if (c.failed) { fprintf(stderr, "gguf: kv[%llu] '%s' value truncated\n",
                                        (unsigned long long)i, k->key); return -1; }
                k->f = (double)rd_f32(c.p); k->i = (int64_t)k->f; c.p += 4; break;
            case BGGUF_UINT64: case BGGUF_INT64:
                bg_need(&c, 8);
                if (c.failed) { fprintf(stderr, "gguf: kv[%llu] '%s' value truncated\n",
                                        (unsigned long long)i, k->key); return -1; }
                k->i = (int64_t)rd_u64(c.p); c.p += 8; break;
            case BGGUF_FLOAT64:
                bg_need(&c, 8);
                if (c.failed) { fprintf(stderr, "gguf: kv[%llu] '%s' value truncated\n",
                                        (unsigned long long)i, k->key); return -1; }
                k->f = rd_f64(c.p); k->i = (int64_t)k->f; c.p += 8; break;
            case BGGUF_STRING: {
                uint64_t slen = 0;
                const char *s = bg_str(&c, &slen);
                if (c.failed) return -1;
                size_t sc = slen < sizeof(k->s) - 1 ? slen : sizeof(k->s) - 1;
                memcpy(k->s, s, sc); k->s[sc] = '\0';
                break;
            }
            case BGGUF_ARRAY: {
                uint32_t et = bg_u32(&c);
                uint64_t n  = bg_u64(&c);
                if (c.failed) return -1;

                /* Element size for the scalar types this model actually uses. */
                uint64_t esz = 0;
                switch (et) {
                    case BGGUF_UINT8: case BGGUF_INT8: case BGGUF_BOOL:      esz = 1; break;
                    case BGGUF_UINT16: case BGGUF_INT16:                     esz = 2; break;
                    case BGGUF_UINT32: case BGGUF_INT32: case BGGUF_FLOAT32: esz = 4; break;
                    case BGGUF_UINT64: case BGGUF_INT64: case BGGUF_FLOAT64: esz = 8; break;
                    case BGGUF_STRING:                                       esz = 0; break;
                    default:
                        fprintf(stderr, "gguf: unsupported array elem type %u (%s)\n",
                                et, k->key);
                        return -1;
                }

                /* The whole array must be present before any element is read:
                 * n is attacker-controlled, so bound it by the bytes left
                 * (which also makes esz*n overflow-proof). */
                if (esz) {
                    uint64_t avail = (uint64_t)(c.end - c.p);
                    if (c.failed || n > avail / esz) {
                        fprintf(stderr, "gguf: kv[%llu] '%s' array truncated "
                                        "(%llu elems, %llu bytes left)\n",
                                (unsigned long long)i, k->key,
                                (unsigned long long)n, (unsigned long long)avail);
                        return -1;
                    }
                }

                /* Keep only the small scalar arrays (rope sections, ssm geometry,
                 * hadamard sign_widths), PLUS prism.hadamard.sign_values which
                 * is 28672 ints and is needed to build the sign vectors. The
                 * 248320-entry tokenizer tables are skipped in place. */
                const int keep_big =
                    (strcmp(k->key, "prism.hadamard.sign_values") == 0) && esz == 4;
                if (esz && (n <= 64 || keep_big)) {
                    if (keep_big) {
                        g->had_sign_values = (int32_t *)malloc((size_t)n * 4);
                        if (g->had_sign_values) {
                            for (uint64_t j = 0; j < n; ++j)
                                g->had_sign_values[j] = (int32_t)rd_u32(c.p + 4 * j);
                            g->had_sign_value_count = (int32_t)n;
                        }
                        c.p += esz * n;
                        k->arr_len = 0;      /* the values live on the gguf now */
                        break;
                    }
                    for (uint64_t j = 0; j < n; ++j) {
                        const uint8_t *e = c.p + esz * j;
                        switch (et) {
                            case BGGUF_UINT8: case BGGUF_BOOL:  k->arr[j] = (int64_t)e[0]; break;
                            case BGGUF_INT8:                    k->arr[j] = (int64_t)(int8_t)e[0]; break;
                            case BGGUF_UINT16:                  k->arr[j] = (int64_t)rd_u16(e); break;
                            case BGGUF_INT16:                   k->arr[j] = (int64_t)(int16_t)rd_u16(e); break;
                            case BGGUF_UINT32:                  k->arr[j] = (int64_t)rd_u32(e); break;
                            case BGGUF_INT32:                   k->arr[j] = (int64_t)rd_i32(e); break;
                            case BGGUF_FLOAT32:                 k->arr[j] = (int64_t)rd_f32(e); break;
                            case BGGUF_UINT64: case BGGUF_INT64: k->arr[j] = (int64_t)rd_u64(e); break;
                            case BGGUF_FLOAT64:                 k->arr[j] = (int64_t)rd_f64(e); break;
                            default: break;
                        }
                    }
                    k->arr_len = (int32_t)n;
                    bg_need(&c, esz * n);
                    c.p += esz * n;
                } else if (esz) {
                    bg_need(&c, esz * n);
                    c.p += esz * n;
                } else {
                    for (uint64_t j = 0; j < n; ++j) {
                        uint64_t sl; bg_str(&c, &sl);
                        if (c.failed) return -1;
                    }
                }
                break;
            }
            default:
                fprintf(stderr, "gguf: unknown kv type %u for '%s'\n", vt, k->key);
                return -1;
        }
        ++wi;
    }
    g->kv_len = wi;

    /* ---- tensor table ---- */
    g->tensors = (bonsai_tensor *)calloc(g->n_tensors ? g->n_tensors : 1,
                                         sizeof(bonsai_tensor));
    if (!g->tensors) return -1;

    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        uint64_t nlen = 0;
        const char *ns = bg_str(&c, &nlen);
        if (c.failed) return -1;
        bonsai_tensor *t = &g->tensors[i];
        size_t nc = nlen < sizeof(t->name) - 1 ? nlen : sizeof(t->name) - 1;
        memcpy(t->name, ns, nc);
        t->name[nc] = '\0';

        t->n_dims = (int32_t)bg_u32(&c);
        if (t->n_dims < 1 || t->n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "gguf: bad n_dims %d for %s\n", t->n_dims, t->name);
            return -1;
        }
        for (int32_t d = 0; d < t->n_dims; ++d) t->dims[d] = bg_u64(&c);
        for (int32_t d = t->n_dims; d < GGUF_MAX_DIMS; ++d) t->dims[d] = 1;
        t->type   = bg_u32(&c);
        t->offset = bg_u64(&c);
        if (c.failed) return -1;

        t->nbytes = bonsai_tensor_nbytes(t->type, t->dims, t->n_dims, &t->n_elements);
        if (t->nbytes == 0) {
            fprintf(stderr, "gguf: unsupported type %u for %s\n", t->type, t->name);
            return -1;
        }
    }
    g->tensor_len = g->n_tensors;

    /* the data region begins at the next `alignment` boundary */
    g->alignment   = kv_int(g, "general.alignment", 32);
    if (g->alignment == 0) g->alignment = 32;
    uint64_t info_end = (uint64_t)(c.p - g->map);
    g->data_offset = (info_end + g->alignment - 1) / g->alignment * g->alignment;

    /* sort for binary search */
    qsort(g->tensors, g->tensor_len, sizeof(bonsai_tensor), tensor_cmp);

    /* Validate the data region boundary. The tensor `offset` fields are relative
     * to data_offset (see gguf.h) and were previously trusted outright: a file
     * cut short, or an absurd offset, loaded "successfully" and handed out
     * pointers past the end of the loaded file. Overflow in
     * data_offset + offset + nbytes must be rejected too, because a wrapped sum
     * can land back inside the file. */
    if (g->tensor_len) {
        if (g->data_offset > g->file_size) {
            fprintf(stderr, "gguf: data offset %llu past EOF %llu\n",
                    (unsigned long long)g->data_offset,
                    (unsigned long long)g->file_size);
            return -1;
        }
        uint64_t room = g->file_size - g->data_offset;
        uint64_t highest = 0;
        for (uint64_t i = 0; i < g->tensor_len; ++i) {
            const bonsai_tensor *t = &g->tensors[i];
            if (t->offset > UINT64_MAX - t->nbytes || t->offset + t->nbytes > room) {
                fprintf(stderr, "gguf: tensor '%s' (offset=%llu nbytes=%llu) "
                                "outside the %llu-byte data region\n", t->name,
                        (unsigned long long)t->offset, (unsigned long long)t->nbytes,
                        (unsigned long long)room);
                return -1;
            }
            if (t->offset + t->nbytes > highest) highest = t->offset + t->nbytes;
        }
        if (highest != room) {
            fprintf(stderr, "gguf: warning: tensor payloads cover %llu of the "
                            "%llu data bytes (slack %lld)\n",
                    (unsigned long long)highest, (unsigned long long)room,
                    (long long)(room - highest));
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* geometry                                                            */
/* ------------------------------------------------------------------ */
static void gguf_read_geometry(bonsai_gguf *g) {
    g->n_layer = (int32_t)kv_int(g, "qwen35.block_count", 0);
    g->n_embd  = (int32_t)kv_int(g, "qwen35.embedding_length", 0);
    g->n_ff    = (int32_t)kv_int(g, "qwen35.feed_forward_length", 0);

    g->n_head    = (int32_t)kv_int(g, "qwen35.attention.head_count", 0);
    g->n_head_kv = (int32_t)kv_int(g, "qwen35.attention.head_count_kv", 0);
    g->head_dim  = (int32_t)kv_int(g, "qwen35.attention.key_length", 0);

    g->n_rot          = (int32_t)kv_int(g, "qwen35.rope.dimension_count", 0);
    g->rope_freq_base = kv_float(g, "qwen35.rope.freq_base", 10000.0);
    g->rms_eps        = kv_float(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-5);
    for (int i = 0; i < 4; ++i) g->rope_sections[i] = 0;
    {
        const bonsai_kv *k = kv_find(g, "qwen35.rope.dimension_sections");
        if (k) for (int32_t i = 0; i < k->arr_len && i < 4; ++i) g->rope_sections[i] = k->arr[i];
    }

    g->ssm_d_conv  = (int32_t)kv_int(g, "qwen35.ssm.conv_kernel", 0);
    g->ssm_d_state = (int32_t)kv_int(g, "qwen35.ssm.state_size", 0);
    g->ssm_d_inner = (int32_t)kv_int(g, "qwen35.ssm.inner_size", 0);
    g->ssm_dt_rank = (int32_t)kv_int(g, "qwen35.ssm.time_step_rank", 0);
    g->ssm_n_group = (int32_t)kv_int(g, "qwen35.ssm.group_count", 0);
    g->full_attn_interval = (int32_t)kv_int(g, "qwen35.full_attention_interval", 4);

    g->had_block = (int32_t)kv_int(g, "prism.hadamard.block_size", 0);
    g->had_sign_width_count = 0;
    {
        const bonsai_kv *k = kv_find(g, "prism.hadamard.sign_widths");
        if (k) {
            for (int32_t i = 0; i < k->arr_len && i < 8; ++i) {
                g->had_sign_widths[i] = k->arr[i];
                ++g->had_sign_width_count;
            }
        }
    }
    /* n_vocab is derived from token_embd.weight after parsing, in
     * bonsai_gguf_load(), because it needs the tensor table. */
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* 64-bit file size / seek                                             */
/*                                                                     */
/* ftell() returns a 32-bit `long` on Windows x64 AND on k1om (both    */
/* have 32-bit long), so it silently fails for a 7.2 GB model and      */
/* returns -1. Use the platform's 64-bit variants instead.             */
/* ------------------------------------------------------------------ */
static int64_t file_size_of(FILE *f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return (int64_t)ftello(f);
#endif
}

static int file_seek_end(FILE *f) {
#ifdef _WIN32
    return _fseeki64(f, 0, SEEK_END);
#else
    return fseeko(f, 0, SEEK_END);
#endif
}

bonsai_gguf *bonsai_gguf_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "gguf: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (file_seek_end(f) != 0) {
        fprintf(stderr, "gguf: seek failed on %s\n", path);
        fclose(f);
        return NULL;
    }
    int64_t sz = file_size_of(f);
    if (sz <= 0) {
        fprintf(stderr, "gguf: bad file size %lld for %s\n",
                (long long)sz, path);
        fclose(f);
        return NULL;
    }
    rewind(f);

    bonsai_gguf *g = (bonsai_gguf *)calloc(1, sizeof(bonsai_gguf));
    if (!g) { fclose(f); return NULL; }
    g->file_size = (uint64_t)sz;
    /* Over-allocate by a small guard and zero it.
     *
     * The IMCI PQ2_0 decoder uses _mm512_extloadunpacklo/hi_epi32, whose
     * footprint spans the three 64-byte-aligned blocks bracketing the payload.
     * Measured on the card against a PROT_NONE guard page, decoding one 34-byte
     * block touches [blk, blk+81] -- i.e. it reads up to 48 bytes past the end of
     * the block, independent of alignment (all 64 byte phases gave the same
     * threshold). The last block of the last row of the last PQ2_0 tensor
     * therefore reads past that tensor.
     *
     * Today that lands inside the following tensor (this model's tensor table
     * happens to end with an F32 blob, and glibc leaves page-rounding slack), so
     * nothing faults. But a re-export that puts a PQ2_0 tensor last, or a file
     * size where (size+16) % 4096 == 0, would read past the allocation and
     * segfault. Reserving the slack here makes the read harmless in all cases.
     *
     * Nothing reads g->map beyond file_size except that decoder. */
    const uint64_t guard = 64;
    g->map = (uint8_t *)malloc((size_t)g->file_size + (size_t)guard);
    if (g->map) memset(g->map + g->file_size, 0, (size_t)guard);
    if (!g->map) {
        fprintf(stderr, "gguf: cannot allocate %.2f GB for the model\n",
                (double)g->file_size / 1e9);
        free(g); fclose(f);
        return NULL;
    }
    size_t got = fread(g->map, 1, g->file_size, f);
    fclose(f);
    if (got != g->file_size) {
        fprintf(stderr, "gguf: short read (%zu of %llu)\n", got,
                (unsigned long long)g->file_size);
        free(g->map); free(g);
        return NULL;
    }

    if (gguf_parse(g) != 0) {
        bonsai_gguf_free(g);
        return NULL;
    }
    gguf_read_geometry(g);

    /* vocab: token_embd.weight dims are {n_embd, n_vocab} in ggml order.
     * gguf_parse() has already sorted the table, so the binary search works. */
    const bonsai_tensor *emb = bonsai_tensor_get(g, "token_embd.weight");
    g->n_vocab = emb ? (int32_t)emb->dims[1] : 0;
    return g;
}

void bonsai_gguf_free(bonsai_gguf *g) {
    if (!g) return;
    free(g->kv);
    free(g->tensors);
    free(g->had_sign_values);
    free(g->map);
    free(g);
}

const bonsai_tensor *bonsai_tensor_get(const bonsai_gguf *g, const char *name) {
    bonsai_tensor key;
    memset(&key, 0, sizeof(key));
    strncpy(key.name, name, sizeof(key.name) - 1);
    return (const bonsai_tensor *)bsearch(&key, g->tensors, g->tensor_len,
                                          sizeof(bonsai_tensor), tensor_cmp);
}

const uint8_t *bonsai_tensor_data(const bonsai_gguf *g, const bonsai_tensor *t) {
    return g->map + g->data_offset + t->offset;
}

int bonsai_gguf_selftest(const char *path) {
    bonsai_gguf *g = bonsai_gguf_load(path);
    if (!g) return -1;

    printf("version    : %u\n", g->version);
    printf("alignment  : %llu\n", (unsigned long long)g->alignment);
    printf("data offset: %llu\n", (unsigned long long)g->data_offset);
    printf("n_kv       : %llu\n", (unsigned long long)g->kv_len);
    printf("n_tensors  : %llu\n", (unsigned long long)g->tensor_len);
    printf("file size  : %llu\n", (unsigned long long)g->file_size);

    printf("\ngeometry:\n");
    printf("  n_layer=%d n_embd=%d n_ff=%d n_vocab=%d\n",
           g->n_layer, g->n_embd, g->n_ff, g->n_vocab);
    printf("  n_head=%d n_head_kv=%d head_dim=%d n_rot=%d\n",
           g->n_head, g->n_head_kv, g->head_dim, g->n_rot);
    printf("  ssm: conv=%d state=%d inner=%d dt_rank=%d group=%d\n",
           g->ssm_d_conv, g->ssm_d_state, g->ssm_d_inner,
           g->ssm_dt_rank, g->ssm_n_group);
    printf("  full_attn_interval=%d had_block=%d\n",
           g->full_attn_interval, g->had_block);
    printf("  rope: base=%g sections=[%lld %lld %lld %lld]\n",
           g->rope_freq_base,
           (long long)g->rope_sections[0], (long long)g->rope_sections[1],
           (long long)g->rope_sections[2], (long long)g->rope_sections[3]);

    /* every tensor must lie inside the data region. Overflow in
     * data_offset + offset + nbytes must be treated as "past EOF": a wrapped
     * end can land back inside the file and hide an absurd offset. */
    uint64_t highest = 0;
    int bad = 0;
    for (uint64_t i = 0; i < g->tensor_len; ++i) {
        const bonsai_tensor *t = &g->tensors[i];
        uint64_t end;
        int overflow = 0;
        if (t->offset > UINT64_MAX - t->nbytes) overflow = 1;
        else {
            uint64_t rel = t->offset + t->nbytes;
            if (rel > UINT64_MAX - g->data_offset) overflow = 1;
            else end = g->data_offset + rel;
        }
        if (overflow) {
            fprintf(stderr, "  !! %s offset+nbytes overflows\n", t->name);
            ++bad;
            continue;
        }
        if (end > highest) highest = end;
        if (end > g->file_size) {
            fprintf(stderr, "  !! %s extends past EOF\n", t->name);
            ++bad;
        }
    }
    printf("\nintegrity:\n");
    printf("  highest tensor end = %llu\n", (unsigned long long)highest);
    printf("  file size          = %llu\n", (unsigned long long)g->file_size);
    printf("  slack              = %lld\n", (long long)g->file_size - (long long)highest);
    printf("  tensors past EOF   = %d\n", bad);

    const bonsai_tensor *t = bonsai_tensor_get(g, "blk.0.attn_qkv.weight");
    if (t) {
        printf("\nprobe blk.0.attn_qkv.weight:\n");
        printf("  dims={%llu, %llu} type=%s nbytes=%llu\n",
               (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
               bonsai_type_name(t->type), (unsigned long long)t->nbytes);
    }
    const bonsai_tensor *c1 = bonsai_tensor_get(g, "blk.0.ssm_conv1d.weight");
    if (c1) {
        printf("probe blk.0.ssm_conv1d.weight: dims={%llu, %llu} type=%s\n",
               (unsigned long long)c1->dims[0], (unsigned long long)c1->dims[1],
               bonsai_type_name(c1->type));
    }

    bonsai_gguf_free(g);
    return bad == 0 ? 0 : -1;
}
