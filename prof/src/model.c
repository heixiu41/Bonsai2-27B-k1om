/*
 * model.c - Bonsai 2 27B (qwen35) forward pass for the Xeon Phi 7120P.
 *
 * The structure deliberately mirrors scripts/bonsai_forward.py, which is the
 * verified oracle: every stage here corresponds to a named stage there, so a
 * divergence can be bisected by dumping the same tensor from both.
 *
 * Ops confirmed against the official llama-eval-callback dump of this model:
 *   - the Hadamard activation transform is  x' = H . (s * x)  applied on the
 *     INPUT side of the 401 folded weights; H is blockwise over 1024, the sign
 *     vector matches the weight's input width.
 *   - the embedding row is inverse-transformed as  s . (H x)  (H first, then
 *     the width-5120 signs).
 *   - there is NO extra Hadamard inside attention (no self_k_rot/self_v_rot).
 *   - data_offset + sum(nbytes) == file_size for this model (no gaps).
 */
/* posix_memalign needs this before ANY system header. k1om uses _mm_malloc
 * instead, so the macro is only needed for the host build.
 * The profiling timers (prof2.h) need CLOCK_MONOTONIC from <time.h>, which also
 * requires this to be set before the first system header, so the k1om build
 * needs it too whenever BONSAI_PROF2 is on. */
#if !defined(__MIC__) || defined(BONSAI_PROF2)
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200809L
#  endif
#endif

#include "model.h"
#include "prof2.h"   /* region timers; compiled out unless -DBONSAI_PROF2 */

/* prof2.h's P2_T() compiles away when the macro is off, but the hand-written
 * probes below call p2_now()/p2_add() directly, so they need guarding. */
#ifdef BONSAI_PROF2
#  define P2_DIRECT 1
#else
#  define P2_DIRECT 0
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Widest folded input (ffn_down). sc.fold_x holds one of these and mm_folded()
 * stages every matmul input through it, so nothing wider may reach it. */
#define BONSAI_FOLD_MAX 17408

/* ================================================================== */
/* scalar helpers                                                      */
/* ================================================================== */
static inline float f_silu(float x)    { return x / (1.0f + expf(-x)); }
static inline float f_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float f_softplus(float x) {
    /* log1p(exp(-|x|)) + max(x,0): stable for large |x| */
    return log1pf(expf(-fabsf(x))) + (x > 0.0f ? x : 0.0f);
}

/* RMSNorm over n elements; w may be NULL */
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    if (w) for (int i = 0; i < n; ++i) o[i] = x[i] * scale * w[i];
    else   for (int i = 0; i < n; ++i) o[i] = x[i] * scale;
}

/* L2 normalisation over n elements (ggml_l2_norm) */
static void l2norm(float *o, const float *x, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    const float inv = (ss > 0.0) ? (1.0f / sqrtf((float)ss)) : 0.0f;
    for (int i = 0; i < n; ++i) o[i] = x[i] * inv;
}

/* ================================================================== */
/* tensor access                                                       */
/* ================================================================== */
static const bonsai_tensor *T(const bonsai_model *m, const char *name) {
    return bonsai_tensor_get(m->g, name);
}

/* Dequantise any supported tensor to fp32 (F32/F16/BF16 direct, PQ2_0 packed).
 * `out` is checked against the capacity of the destination: the buffers here are
 * sized from the model geometry while `t->n_elements` comes from the file, and a
 * crafted file can make the two disagree. Returns 0 on success, -1 on a missing
 * tensor, an unsupported type, or a destination that is too small. */
static int dequant_to_f32(const bonsai_model *m, const char *name, float *out,
                          size_t cap) {
    const bonsai_tensor *t = T(m, name);
    if (!t) {
        fprintf(stderr, "dequant: missing tensor %s\n", name);
        return -1;
    }
    const uint8_t *p = bonsai_tensor_data(m->g, t);
    const uint64_t n = t->n_elements;
    if (n > (uint64_t)cap) {
        fprintf(stderr, "dequant: %s holds %llu elements, destination %llu\n",
                name, (unsigned long long)n, (unsigned long long)cap);
        return -1;
    }

    switch (t->type) {
        case GGML_TYPE_F32:
            memcpy(out, p, (size_t)n * 4);
            return 0;
        case GGML_TYPE_F16:
            for (uint64_t i = 0; i < n; ++i)
                out[i] = bonsai_fp16_to_fp32((uint16_t)(p[2*i] | (p[2*i+1] << 8)));
            return 0;
        case GGML_TYPE_BF16:
            for (uint64_t i = 0; i < n; ++i)
                out[i] = bonsai_bf16_to_fp32((uint16_t)(p[2*i] | (p[2*i+1] << 8)));
            return 0;
        case GGML_TYPE_PQ2_0:
            pq2_0_dequant_matrix(p, out, (size_t)n);
            return 0;
        default:
            fprintf(stderr, "dequant: unsupported type %u for %s\n", t->type, name);
            return -1;
    }
}

/* ================================================================== */
/* folded-weight matmul                                                */
/*                                                                     */
/*   y[o] = sum_i W[o][i] * had(x)[i]                                  */
/*                                                                     */
/* When `signs` is non-NULL the input is transformed in place into        */
/* sc.fold_x (the Hadamard transform is not free, so we never touch the  */
/* caller's buffer). Weights stay in PQ2_0 form and are decoded 128 at a */
/* time.                                                                */
/* ================================================================== */
/* ------------------------------------------------------------------ */
/* Portable 64-byte aligned float allocation                           */
/*                                                                     */
/* Needed because the IMCI kernels require _mm512_load_ps alignment.   */
/* Deliberately not aligned_alloc / posix_memalign: Zig's libc         */
/* declares neither without feature-test macros, and a silent fallback */
/* would silently reintroduce misaligned vector loads.                 */
/* ------------------------------------------------------------------ */
static float *aligned_f32(size_t count, size_t align) {
    void *raw = malloc(count * sizeof(float) + align);
    if (!raw) return NULL;
    const uintptr_t base = (uintptr_t)raw + (uintptr_t)align;
    float *p = (float *)(base & ~(uintptr_t)(align - 1));
    memset(p, 0, count * sizeof(float));
    ((void **)p)[-1] = raw;
    return p;
}
static void aligned_f32_free(float *p) {
    if (p) free(((void **)p)[-1]);
}

/* Returns 0 on success; -1 when the weight is missing, is not PQ2_0, or does
 * not match the caller's geometry. On failure `y` is left untouched, so the
 * caller MUST abort the token: returning silently used to leave `y` holding the
 * previous token's values while bonsai_forward() still reported success.
 *
 * The tensor type is checked explicitly: gguf.h gives GGML_TYPE_PTQ1_0 (143)
 * the same 34-byte block size as PQ2_0, so such a tensor loads cleanly and
 * would otherwise be decoded with the PQ2_0 code map, producing garbage with no
 * error at all (dequant_to_f32() rejects it, but the mm path used not to look). */
static int mm_folded(bonsai_model *m, float *y, const float *x,
                     const char *wname, const float *signs, int n, int d_cap) {
    const bonsai_tensor *t = T(m, wname);
    if (!t) {
        fprintf(stderr, "mm_folded: missing weight %s\n", wname);
        return -1;
    }
    if (t->type != GGML_TYPE_PQ2_0) {
        fprintf(stderr, "mm_folded: %s is %s (%u), expected PQ2_0 (%u)\n",
                wname, bonsai_type_name(t->type), t->type,
                (unsigned)GGML_TYPE_PQ2_0);
        return -1;
    }
    if (n <= 0 || t->n_dims < 2 || n > (int)BONSAI_FOLD_MAX ||
        t->dims[0] != (uint64_t)n || t->dims[1] > (uint64_t)d_cap) {
        fprintf(stderr, "mm_folded: %s shape mismatch (dims {%llu,%llu}, "
                        "caller n=%d cap=%d)\n", wname,
                (unsigned long long)t->dims[0],
                (unsigned long long)t->dims[1], n, d_cap);
        return -1;
    }
    const uint8_t *w = bonsai_tensor_data(m->g, t);
    const int d = (int)t->dims[1];      /* ggml ne1 = output dim */

    if (signs && (m->had_block <= 0 || n % m->had_block)) {
        /* hadamard_forward() steps base += block without checking n % block and
         * would write past the end of the input for a non-multiple. */
        fprintf(stderr, "mm_folded: %s input width %d is not a multiple of "
                        "had_block %d\n", wname, n, m->had_block);
        return -1;
    }

    /* Every input is staged through the 64-byte-aligned fold_x before it
     * reaches the IMCI kernel: pq2_0_matvec_knc() faults on a merely
     * 16-byte-aligned x (calloc default), and only mm_folded knows whether the
     * caller's buffer qualifies. This used to be true only for signs != NULL,
     * which left mm_plain() -- and any other signs==NULL caller -- as an
     * alignment landmine that could only fire on the card. */
    const float *xin = m->sc.fold_x;
    if (signs) {
        memcpy(m->sc.fold_x, x, (size_t)n * sizeof(float));
        for (int i = 0; i < n; ++i) m->sc.fold_x[i] *= signs[i];
        if (m->debug_want && strcmp(m->debug_want, "signed") == 0 && !m->debug_len) {
            int c = n < m->debug_cap ? n : m->debug_cap;
            memcpy(m->debug, m->sc.fold_x, (size_t)c * sizeof(float));
            m->debug_len = c;
        }
        /* hadamard_forward now reports misuse (block<=0 or block !| n) instead
         * of silently writing past the buffer. bonsai_model_open validates the
         * widths up front, so a failure here means the two disagree. */
        const double _p2h = p2_now();
        if (hadamard_forward(m->sc.fold_x, NULL, n, m->had_block, 0) != 0) {
            fprintf(stderr,
                    "mm_folded: hadamard_forward rejected n=%d block=%d\n",
                    n, m->had_block);
            return -1;
        }
        if (p2_master()) p2_add(P2_MM_HADAMARD, p2_now() - _p2h);
        if (m->debug_want && strcmp(m->debug_want, "hx") == 0 && !m->debug_len
            && n == 5120) {
            int c = n < m->debug_cap ? n : m->debug_cap;
            memcpy(m->debug, m->sc.fold_x, (size_t)c * sizeof(float));
            m->debug_len = c;
        }
    } else {
        memcpy(m->sc.fold_x, x, (size_t)n * sizeof(float));
    }

#ifdef __MIC__
    const double _p2m = p2_now();
    pq2_0_matvec_knc(y, xin, w, n, d);
    if (p2_master()) p2_add(P2_MM_MATVEC, p2_now() - _p2m);
#else
    pq2_0_matvec_ref(y, xin, w, n, d);
#endif
    return 0;
}

/* Plain fp32 matmul against a tensor we previously dequantised into RAM.
 * Used for the small per-layer tables (alpha/beta/conv1d) and the output head. */
static void matvec_f32(float *y, const float *x, const float *w, int n, int d) {
    for (int o = 0; o < d; ++o) {
        const float *row = w + (size_t)o * n;
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

/* ================================================================== */
/* M-RoPE over the first n_rot dims, NeoX pairing (i, i+n_rot/2)        */
/*                                                                     */
/* theta_i = pos * freq_base^(-2i/n_rot), i in [0, n_rot/2)            */
/* Verified against ggml-cpu/ops.cpp rotate_pairs() with               */
/* n_offset = n_dims/2 and scale = 2.                                  */
/* ================================================================== */
static void rope_apply(float *x, int n_head, int head_dim, int n_rot,
                       int pos, double freq_base,
                       const float *cos_t, const float *sin_t) {
    const int half = n_rot / 2;
    (void)freq_base;
    for (int h = 0; h < n_head; ++h) {
        float *v = x + (size_t)h * head_dim;
        for (int i = 0; i < half; ++i) {
            const float c = cos_t[i], s = sin_t[i];
            const float a = v[i];
            const float b = v[i + half];
            v[i]        = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }
}

static void rope_tables(int n_rot, int pos, double freq_base,
                        float *cos_t, float *sin_t) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const double theta = (double)pos * pow(freq_base, -2.0 * i / n_rot);
        cos_t[i] = (float)cos(theta);
        sin_t[i] = (float)sin(theta);
    }
}

/* ================================================================== */
/* lifecycle                                                           */
/* ================================================================== */
static int read_sign_vector(const bonsai_gguf *g, int width, float *out) {
    /* prism.hadamard.sign_values is cached by the loader; had_sign_widths[]
     * partitions it in order. */
    if (!g->had_sign_values || g->had_sign_value_count <= 0) return -1;
    uint64_t off = 0;
    for (int32_t i = 0; i < g->had_sign_width_count; ++i) {
        if (g->had_sign_widths[i] == width) break;
        off += (uint64_t)g->had_sign_widths[i];
    }
    if (off + (uint64_t)width > (uint64_t)g->had_sign_value_count) return -1;
    for (int i = 0; i < width; ++i)
        out[i] = (float)g->had_sign_values[off + i];
    return 0;
}

bonsai_model *bonsai_model_open(const char *gguf_path, int32_t ctx) {
    bonsai_gguf *g = bonsai_gguf_load(gguf_path);
    if (!g) return NULL;

    bonsai_model *m = (bonsai_model *)calloc(1, sizeof(bonsai_model));
    if (!m) { bonsai_gguf_free(g); return NULL; }
    m->g = g;

    m->n_layer = g->n_layer;
    m->n_embd  = g->n_embd;
    m->n_ff    = g->n_ff;
    m->n_vocab = g->n_vocab;
    m->n_head    = g->n_head;
    m->n_head_kv = g->n_head_kv;
    m->head_dim  = g->head_dim;
    m->n_rot     = g->n_rot;
    m->rms_eps   = (float)g->rms_eps;
    m->rope_freq_base = g->rope_freq_base;
    m->full_attn_interval = g->full_attn_interval;
    m->had_block = g->had_block;
    m->ssm_d_conv  = g->ssm_d_conv;
    m->ssm_d_state = g->ssm_d_state;
    m->ssm_inner   = g->ssm_d_inner;
    m->ssm_dt_rank = g->ssm_dt_rank;
    m->ssm_group   = g->ssm_n_group;

    /* Every product and BOTH divisions are computed in int64 and validated
     * BEFORE they are used. The previous revision evaluated
     * `value_dim / ssm_dt_rank` here (and `rep = vh / kh` in block_gdn) before
     * any check covered those fields, so a crafted GGUF died with SIGFPE
     * instead of reporting a bad geometry. */
    const int64_t ds  = m->ssm_d_state;
    const int64_t grp = m->ssm_group;
    const int64_t dtr = m->ssm_dt_rank;
    const int64_t g_key_dim   = ds * grp;
    const int64_t g_value_dim = ds * dtr;
    const int64_t g_conv_dim  = g_key_dim * 2 + g_value_dim;
    const int64_t g_qdim      = (int64_t)m->n_head * m->head_dim;

    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_ff <= 0 || m->n_vocab <= 0 ||
        m->n_head <= 0 || m->n_head_kv <= 0 || m->head_dim <= 0 ||
        m->n_rot <= 0 || m->n_rot > m->head_dim ||
        m->n_head % m->n_head_kv != 0 ||          /* block_attn: rep = nh/nkv */
        m->full_attn_interval <= 0 ||             /* used as a divisor later  */
        ds <= 0 || grp <= 0 || dtr <= 0 ||
        dtr % grp != 0 ||                         /* block_gdn: rep = dtr/grp */
        g_key_dim <= 0 || g_key_dim > INT32_MAX ||
        g_value_dim <= 0 || g_value_dim > INT32_MAX ||
        g_conv_dim <= 0 || g_conv_dim > INT32_MAX ||
        g_qdim <= 0 || g_qdim > INT32_MAX / 2 ||  /* qg/attn_out are 2*qdim */
        m->ssm_d_conv < 2) {                      /* conv_hist holds conv_k-1 */
        fprintf(stderr, "model: bad geometry from GGUF\n");
        bonsai_model_close(m);
        return NULL;
    }
    m->key_dim    = (int32_t)g_key_dim;           /* 2048 */
    m->value_dim  = (int32_t)g_value_dim;         /* 6144 */
    m->conv_dim   = (int32_t)g_conv_dim;          /* 10240 */
    m->head_v_dim = (int32_t)(g_value_dim / dtr); /* 128, dtr > 0 (checked) */
    /* context: default to something that fits comfortably (the reference caps
     * at 2048 on this hardware class). */
    if (ctx <= 0) ctx = 2048;
    m->ctx = ctx;

    /* ---- Hadamard sign vectors ---- */
    if (m->had_block > 0 && g->had_sign_width_count > 0) {
        m->signs_5120  = (float *)malloc(sizeof(float) * 5120);
        m->signs_6144  = (float *)malloc(sizeof(float) * 6144);
        m->signs_17408 = (float *)malloc(sizeof(float) * 17408);
        if (!m->signs_5120 || !m->signs_6144 || !m->signs_17408 ||
            read_sign_vector(g, 5120,  m->signs_5120)  != 0 ||
            read_sign_vector(g, 6144,  m->signs_6144)  != 0 ||
            read_sign_vector(g, 17408, m->signs_17408) != 0) {
            fprintf(stderr, "model: cannot read Hadamard sign vectors\n");
            bonsai_model_close(m);
            return NULL;
        }
    } else {
        fprintf(stderr, "model: no Hadamard metadata; model would be wrong\n");
        bonsai_model_close(m);
        return NULL;
    }

    /* Every width that reaches hadamard_forward() must be an exact multiple of
     * had_block: that function does `for (base = 0; base < n; base += block)`
     * with no n % block check and writes a whole block per iteration, so a
     * remainder overruns the buffer by up to block-1 floats. sc.fold_x is sized
     * for the widest folded input, so nothing wider may be staged in it either.
     * (model.c is not allowed to fix pq2_0.h; this is the caller-side guard.) */
    if (m->n_embd % m->had_block || m->n_ff % m->had_block ||
        g_value_dim % m->had_block || g_qdim % m->had_block ||
        m->n_embd > BONSAI_FOLD_MAX || m->n_ff > BONSAI_FOLD_MAX ||
        g_value_dim > BONSAI_FOLD_MAX || g_qdim > BONSAI_FOLD_MAX) {
        fprintf(stderr, "model: input widths incompatible with had_block %d "
                        "(embd %d ffn %d value %d q %d)\n",
                m->had_block, m->n_embd, m->n_ff,
                (int)g_value_dim, (int)g_qdim);
        bonsai_model_close(m);
        return NULL;
    }

    /* ---- recurrent state + scratch ---- */
    const size_t L = (size_t)m->n_layer;
    m->st.ssm_state = (float *)calloc(L * m->ssm_dt_rank * m->ssm_d_state *
                                      m->ssm_d_state, sizeof(float));
    m->st.conv_hist = (float *)calloc(L * (m->ssm_d_conv - 1) * m->conv_dim,
                                      sizeof(float));
    m->st.kv_cache  = (float *)calloc(L * (size_t)ctx * 2 * m->n_head_kv *
                                      m->head_dim, sizeof(float));
    m->st.kv_len = 0;

    bonsai_scratch *s = &m->sc;
    const int n_embd = m->n_embd, conv_dim = m->conv_dim, value_dim = m->value_dim;
    const int n_ff = m->n_ff, n_head = m->n_head, head_dim = m->head_dim;
    const int vh = m->ssm_dt_rank, st = m->ssm_d_state;

    s->x        = (float *)calloc(n_embd, sizeof(float));
    s->xb       = (float *)calloc(n_embd, sizeof(float));
    s->xb2      = (float *)calloc(n_embd, sizeof(float));
    s->qkv      = (float *)calloc(conv_dim, sizeof(float));
    s->z        = (float *)calloc(value_dim, sizeof(float));
    s->conv_out = (float *)calloc(conv_dim, sizeof(float));
    s->q        = (float *)calloc((size_t)vh * st, sizeof(float));
    s->k        = (float *)calloc((size_t)vh * st, sizeof(float));
    s->v        = (float *)calloc((size_t)vh * st, sizeof(float));
    s->o        = (float *)calloc((size_t)vh * st, sizeof(float));
    s->delta    = (float *)calloc((size_t)vh * st, sizeof(float));
    s->beta_out = (float *)calloc((size_t)2 * vh, sizeof(float));
    s->attn_out = (float *)calloc(value_dim, sizeof(float));
    s->hb       = (float *)calloc(n_ff, sizeof(float));
    s->hb2      = (float *)calloc(n_ff, sizeof(float));
    s->logits   = (float *)calloc(m->n_vocab, sizeof(float));
    s->qg       = (float *)calloc((size_t)2 * n_head * head_dim, sizeof(float));
    s->q_full   = (float *)calloc((size_t)n_head * head_dim, sizeof(float));
    s->attn_gate = (float *)calloc((size_t)n_head * head_dim, sizeof(float));
    s->k_full   = (float *)calloc((size_t)m->n_head_kv * head_dim, sizeof(float));
    s->v_full   = (float *)calloc((size_t)m->n_head_kv * head_dim, sizeof(float));
    s->att      = (float *)calloc((size_t)n_head * ctx, sizeof(float));
    s->rope_cos = (float *)calloc(m->n_rot / 2, sizeof(float));
    s->rope_sin = (float *)calloc(m->n_rot / 2, sizeof(float));
    s->scores   = (float *)calloc(ctx, sizeof(float));
    /* 64-byte aligned: the IMCI kernels in pq2_0.h feed this buffer to
     * _mm512_load_ps, which faults unless the address is 64-byte aligned.
     * calloc only guarantees 16 (glibc), so it must not be used here.
     *
     * The allocation is done by hand rather than with aligned_alloc or
     * posix_memalign: neither is declared by every libc without a
     * feature-test macro (Zig's libc declares neither under -std=c11 nor with
     * _POSIX_C_SOURCE), and a silent fallback to a 16-byte-aligned buffer
     * would reintroduce exactly the alignment fault this code exists to
     * avoid. malloc-overallocate-and-round works on every target. */
    s->fold_x = aligned_f32(BONSAI_FOLD_MAX, 64);

    /* per-layer fp32 tables, dequantised once per layer on demand */
    s->ssm_alpha = (float *)calloc((size_t)n_embd * vh, sizeof(float));
    s->ssm_beta  = (float *)calloc((size_t)n_embd * vh, sizeof(float));
    s->ssm_a     = (float *)calloc(vh, sizeof(float));
    s->ssm_dt    = (float *)calloc(vh, sizeof(float));
    s->ssm_norm  = (float *)calloc(m->head_v_dim, sizeof(float));
    s->conv1d    = (float *)calloc((size_t)m->ssm_d_conv * conv_dim, sizeof(float));
    s->norm_w    = (float *)calloc(n_embd, sizeof(float));
    s->attn_qn   = (float *)calloc(head_dim, sizeof(float));
    s->attn_kn   = (float *)calloc(head_dim, sizeof(float));

    /* Every allocation above is checked: on a 16 GB card holding a 7.2 GB model
     * OOM is realistic, and the previous revision checked 5 of ~25 buffers, so
     * a failed calloc surfaced much later as a NULL dereference instead of this
     * clean error. All failures still funnel into bonsai_model_close(). */
    if (!m->st.ssm_state || !m->st.conv_hist || !m->st.kv_cache ||
        !s->x || !s->xb || !s->xb2 || !s->qkv || !s->z || !s->conv_out ||
        !s->q || !s->k || !s->v || !s->o || !s->delta || !s->beta_out ||
        !s->attn_out || !s->hb || !s->hb2 || !s->logits || !s->qg ||
        !s->q_full || !s->attn_gate || !s->k_full || !s->v_full || !s->att ||
        !s->rope_cos || !s->rope_sin || !s->scores || !s->fold_x ||
        !s->ssm_alpha || !s->ssm_beta || !s->ssm_a || !s->ssm_dt ||
        !s->ssm_norm || !s->conv1d || !s->norm_w || !s->attn_qn ||
        !s->attn_kn) {
        fprintf(stderr, "model: allocation failed\n");
        bonsai_model_close(m);
        return NULL;
    }
    return m;
}

void bonsai_model_close(bonsai_model *m) {
    if (!m) return;
    free(m->st.ssm_state); free(m->st.conv_hist); free(m->st.kv_cache);
    bonsai_scratch *s = &m->sc;
    free(s->x); free(s->xb); free(s->xb2); free(s->qkv); free(s->z);
    free(s->conv_out); free(s->q); free(s->k); free(s->v); free(s->delta);
    free(s->o); free(s->beta_out);
    free(s->attn_out); free(s->hb); free(s->hb2); free(s->logits);
    free(s->qg); free(s->q_full); free(s->attn_gate);
    free(s->k_full); free(s->v_full);
    free(s->att); free(s->rope_cos); free(s->rope_sin); free(s->scores);
    aligned_f32_free(s->fold_x);
    free(s->ssm_alpha); free(s->ssm_beta); free(s->ssm_a);
    free(s->ssm_dt); free(s->ssm_norm); free(s->conv1d);
    free(s->norm_w); free(s->attn_qn); free(s->attn_kn);
    free(m->signs_5120); free(m->signs_6144); free(m->signs_17408);
    bonsai_gguf_free(m->g);
    free(m);
}

uint64_t bonsai_model_runtime_bytes(const bonsai_model *m) {
    uint64_t b = 0;
    b += (uint64_t)m->n_layer * m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state * 4;
    b += (uint64_t)m->n_layer * (m->ssm_d_conv - 1) * m->conv_dim * 4;
    b += (uint64_t)m->n_layer * m->ctx * 2 * m->n_head_kv * m->head_dim * 4;
    b += (uint64_t)m->n_embd * 3 * 4;
    b += (uint64_t)m->conv_dim * 2 * 4;
    b += (uint64_t)m->value_dim * 2 * 4;
    b += (uint64_t)m->ssm_dt_rank * m->ssm_d_state * 4 * 4;
    b += (uint64_t)m->n_ff * 2 * 4;
    b += (uint64_t)m->n_vocab * 4;
    b += (uint64_t)2 * m->n_head * m->head_dim * 4;
    b += (uint64_t)m->n_head * m->ctx * 4;
    b += 17408u * 4;
    b += (uint64_t)m->n_embd * m->ssm_dt_rank * 2 * 4;
    b += (uint64_t)m->ssm_d_conv * m->conv_dim * 4;
    b += 5120u * 4 + 6144u * 4 + 17408u * 4;
    return b;
}

void bonsai_model_describe(const bonsai_model *m) {
    printf("Bonsai 2 27B (qwen35) on k1om\n");
    printf("  layers %d  embd %d  ffn %d  vocab %d\n",
           m->n_layer, m->n_embd, m->n_ff, m->n_vocab);
    printf("  attention: heads %d / kv %d  head_dim %d  n_rot %d  rope_base %g\n",
           m->n_head, m->n_head_kv, m->head_dim, m->n_rot, m->rope_freq_base);
    printf("  GDN: key_dim %d value_dim %d conv_dim %d state %d heads %d conv_k %d\n",
           m->key_dim, m->value_dim, m->conv_dim, m->ssm_d_state,
           m->ssm_dt_rank, m->ssm_d_conv);
    printf("  full attention every %d blocks (first at index %d)\n",
           m->full_attn_interval, m->full_attn_interval - 1);
    printf("  context allocated: %d\n", m->ctx);
    printf("  weights (PQ2_0, on disk) : %.2f GB\n",
           (double)m->g->file_size / 1e9);
    printf("  runtime state + scratch  : %.3f GB\n",
           (double)bonsai_model_runtime_bytes(m) / 1e9);
}

void bonsai_reset(bonsai_model *m) {
    const size_t L = (size_t)m->n_layer;
    memset(m->st.ssm_state, 0, L * m->ssm_dt_rank * m->ssm_d_state *
                               m->ssm_d_state * sizeof(float));
    memset(m->st.conv_hist, 0, L * (m->ssm_d_conv - 1) * m->conv_dim *
                               sizeof(float));
    memset(m->st.kv_cache, 0, L * (size_t)m->ctx * 2 * m->n_head_kv *
                              m->head_dim * sizeof(float));
    m->st.kv_len = 0;
}

/* Optional debug tap: copy v[0..n) into m->debug when the requested stage of
 * the requested block is being executed. Used by test_forward.c to bisect
 * against the official dump; costs one strcmp per block when disabled. */
static void debug_tap(bonsai_model *m, int il, const char *stage,
                      const float *v, int n) {
    if (!m->debug || !m->debug_want) return;
    if (il != m->debug_layer) return;
    if (strcmp(m->debug_want, stage) != 0) return;
    int c = n < m->debug_cap ? n : m->debug_cap;
    memcpy(m->debug, v, (size_t)c * sizeof(float));
    m->debug_len = c;
}

/* ================================================================== */
/* the two block types                                                 */
/* ================================================================== */

/* Gated Delta Net block (48 of the 64 blocks).
 *
 * Mirrors bonsai_forward.py::layer_ssm exactly:
 *   xb   = RMSNorm(x, attn_norm)
 *   qkv  = W_qkv  @ had(xb)          [conv_dim]
 *   z    = W_gate @ had(xb)          [value_dim]
 *   beta = sigmoid(W_beta @ xb)      [n_v_heads]      (BF16 weight)
 *   alpha= softplus(W_alpha @ xb + ssm_dt)            (BF16 weight)
 *   g    = alpha * ssm_a             (ssm_a is already -A_log.exp())
 *   conv = silu( depthwise_causal_conv(qkv, kernel=4) )
 *   q,k  = split(conv[0:2*key_dim]) then ONE shared L2 normalisation
 *   v    = conv[2*key_dim : ]
 *   k,q repeated x3 to line up with the 48 value heads
 *   S <- S*exp(g);  sk = S^T k;  d = beta*(v - sk);  S += k (x) d;  o = S^T q
 *   out  = W_ssm_out @ had( silu(z) * RMSNorm_perhead(o, ssm_norm) )
 */
static int block_gdn(bonsai_model *m, int il) {
    bonsai_scratch *s = &m->sc;
    const int n_embd  = m->n_embd;
    const int key_dim = m->key_dim, value_dim = m->value_dim;
    const int conv_dim = m->conv_dim;
    const int st = m->ssm_d_state, vh = m->ssm_dt_rank, kh = m->ssm_group;
    const int ck = m->ssm_d_conv;
    /* Validate the fields used as divisors before dividing (bonsai_model_open()
     * rejects them too; this keeps the block safe when driven directly). */
    if (vh <= 0 || kh <= 0 || vh % kh || st <= 0 || key_dim <= 0 ||
        value_dim <= 0 || conv_dim <= 0 || ck < 1 ||
        n_embd <= 0 || n_embd > BONSAI_FOLD_MAX) {
        fprintf(stderr, "block_gdn: bad geometry (vh=%d kh=%d state=%d conv_k=%d)\n",
                vh, kh, st, ck);
        return -1;
    }
    const int rep = vh / kh;
    char nm[96];

    snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
    if (dequant_to_f32(m, nm, s->norm_w, (size_t)n_embd) != 0) return -1;
    rmsnorm(s->xb, s->x, s->norm_w, n_embd, m->rms_eps);
    if (!m->debug_len) debug_tap(m, il, "attn_norm", s->xb, n_embd);

    snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", il);
    if (mm_folded(m, s->qkv, s->xb, nm, m->signs_5120, n_embd, conv_dim) != 0)
        return -1;
    debug_tap(m, il, "qkv", s->qkv, conv_dim);

    snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", il);
    if (mm_folded(m, s->z, s->xb, nm, m->signs_5120, n_embd, value_dim) != 0)
        return -1;

    snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", il);
    if (dequant_to_f32(m, nm, s->ssm_beta, (size_t)n_embd * vh) != 0) return -1;
    matvec_f32(s->beta_out, s->xb, s->ssm_beta, n_embd, vh);

    snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", il);
    if (dequant_to_f32(m, nm, s->ssm_alpha, (size_t)n_embd * vh) != 0) return -1;
    matvec_f32(s->beta_out + vh, s->xb, s->ssm_alpha, n_embd, vh);

    snprintf(nm, sizeof(nm), "blk.%d.ssm_a", il);
    if (dequant_to_f32(m, nm, s->ssm_a, (size_t)vh) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", il);
    if (dequant_to_f32(m, nm, s->ssm_dt, (size_t)vh) != 0) return -1;

    float *beta = s->beta_out;
    float *g    = s->beta_out + vh;      /* reuse: alpha -> g */
    for (int h = 0; h < vh; ++h) {
        beta[h] = f_sigmoid(beta[h]);
        g[h] = f_softplus(g[h] + s->ssm_dt[h]) * s->ssm_a[h];
    }

    /* ---- causal depthwise conv over the conv_dim channels, kernel 4 ---- */
    snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", il);
    if (dequant_to_f32(m, nm, s->conv1d, (size_t)ck * conv_dim) != 0) return -1;
    /*
     * On-disk layout, verified against the official dump: the tensor claims
     * dims={4, 10240} but the data is stored CHANNEL-MAJOR with the kernel tap
     * fastest, i.e. flat[c*conv_k + k]. Indexing it as [k*conv_dim + c] reads
     * the transposed weights and yields a completely different convolution
     * (the first tap of the wrong channel).
     */
    {
        const float *w = s->conv1d;                 /* w[c*ck + k] */
        float *hist = m->st.conv_hist + (size_t)il * (ck - 1) * conv_dim;
        const double _p2c = p2_now();
        /* Channels are independent (each writes only conv_out[c] and reads
         * hist[c], qkv[c], w[c*ck..]) -> parallel. Measured cost was 3.2% of a
         * token and thread-flat, i.e. serial. The history SHIFT below stays
         * serial: its memcpy regions overlap deliberately. */
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < conv_dim; ++c) {
            const float *wc = w + (size_t)c * ck;
            float acc = 0.0f;
            for (int k = 0; k < ck - 1; ++k)          /* oldest .. second-newest */
                acc += wc[k] * hist[(size_t)k * conv_dim + c];
            acc += wc[ck - 1] * s->qkv[c];            /* the current token */
            s->conv_out[c] = f_silu(acc);
        }
        /* conv_hist has (ck-1) slots; with ck == 1 there is no history at all,
         * and `(size_t)(ck - 2) * conv_dim` would be (size_t)-1 * conv_dim, i.e.
         * a conv_dim-float memcpy BEFORE the buffer (calloc(0) returns non-NULL,
         * so the allocation check cannot catch it). bonsai_model_open() rejects
         * ssm_d_conv < 2; this is the matching guard for direct callers. */
        if (ck >= 2) {
            for (int k = 0; k < ck - 2; ++k)
                memcpy(hist + (size_t)k * conv_dim, hist + (size_t)(k + 1) * conv_dim,
                       (size_t)conv_dim * sizeof(float));
            memcpy(hist + (size_t)(ck - 2) * conv_dim, s->qkv,
                   (size_t)conv_dim * sizeof(float));
        }
        debug_tap(m, il, "conv", s->conv_out, conv_dim);
        if (p2_master()) p2_add(P2_GDN_CONV, p2_now() - _p2c);
    }

    /* ---- L2 normalisation: ONE SCALE PER HEAD --------------------------
     * The official graph is a single op over the joint view:
     *     qk_conv_l2-0 = L2_NORM(conv_output_silu-0 (view){128, 32, 1, 1})
     * but ggml_compute_forward_l2_norm_f32 (ggml-cpu/ops.cpp:4195-4218)
     * computes ONE scale per row over ne00, i.e. 32 independent norms of 128
     * (16 q heads + 16 k heads), each head normalised on its own.  The numpy
     * oracle is identical (bonsai_forward.py:216 l2norm(concat([q,k]), axis=-1)
     * == per-head l2norm, max|diff| = 0.0).
     *
     * The code below used to call l2norm() once over the whole 2048-element q
     * block and once over k -- ONE shared scalar for 16 heads, which is neither
     * the op nor the oracle.  On the real block-0 conv output (token 760) the
     * per-head norms spread 10.0x for q and 18.8x for k, so individual heads
     * were mis-scaled by up to 94% (q) / 97% (k). */
    /* Per-head and independent: head h normalises q[h*st .. ] and k[h*st .. ]
     * only. Measured thread-flat (serial) before this pragma. */
    const double _p2l = p2_now();
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < kh; ++h) {
        l2norm(s->q + (size_t)h * st, s->conv_out + (size_t)h * st, st);
        l2norm(s->k + (size_t)h * st,
               s->conv_out + key_dim + (size_t)h * st, st);
    }
    if (p2_master()) p2_add(P2_GDN_L2, p2_now() - _p2l);

    /* v is not normalised */
    memcpy(s->v, s->conv_out + 2 * key_dim, (size_t)value_dim * sizeof(float));

    /* ---- DeltaNet recurrence, per value head ----
     *
     * The state buffer is stored TRANSPOSED with respect to the oracle's S:
     *     Sh[a*st + i]  ==  S_oracle[i][a]        (a = value, i = key)
     * Nothing outside this loop ever reads ssm_state, so the layout is a pure
     * internal convention and any SELF-CONSISTENT choice gives the same output.
     *
     * Oracle form (bonsai_forward.py:235-239), S[i][j] with i = key, j = value:
     *   S *= exp(g)
     *   sk[j] = sum_i S[i][j] * k[i]
     *   d[j]  = beta * (v[j] - sk[j])
     *   S[i][j] += k[i] * d[j]
     *   o[j]  = (1/sqrt(state)) * sum_i S[i][j] * q[i]
     *
     * This loop contracts the FIRST C index (a); with the transposition above
     * that IS sum_i S[i][j]k[i].  The previous revision contracted the SECOND C
     * index (row[j]*khv[j]), which is the same operator written in the opposite
     * layout convention: the two differ exactly by Sh_new == Sh_old^T.
     *
     * MEASURED, not argued (D:\axistest\axis_h5.c + axis_h5.py):
     *   max|loop_old(S^T) - loop_new(S)| = 0.000e+00   (bit-identical)
     *   over 5 tokens started from the all-zero state that bonsai_reset()
     *   installs -- the only initial condition this program ever uses:
     *     max|o_old - o_new| = 0.000e+00 at every token, 0/40 floats differ
     *   end-to-end: linking the pre-change D:\zigtest\model.o produces an output
     *   file byte-identical to the post-change one (same 1077.1537, same SHA256).
     * So this axis change can never move a number in either direction, and
     * neither shape is a bug. */
    const float qscale = 1.0f / sqrtf((float)st);
    float *S = m->st.ssm_state + (size_t)il * vh * st * st;
    /* THE 70% BOTTLENECK. Measured on the card (docs/K1OM_TIME_BUDGET.md): this
     * recurrence was 22.0 s of a 31.4 s nine-token run -- 70.2% of wall clock --
     * and its cost was FLAT from 1 to 244 threads, i.e. it was pure serial work
     * while the matvec next to it scaled from 170 s to 1.7 s.
     *
     * Cause of the slowness: Sh[a*st + i] walks `st` rows that are 512 bytes
     * apart, which ICC turns into microcoded vgatherdps in retry loops
     * (annotated "Latency 45"), the worst instruction class on KNC. Arithmetic
     * says ~15 cycles per FMA that a 16-wide unit would do in one.
     *
     * The 48 value heads are provably independent: head h touches only its own
     * S slice, reads k/q/v/beta/g read-only, and writes only its own o slice.
     * A/B ablation with this one pragma added: recurrence 22.013 s -> 0.578 s
     * (38x), engine 0.291 -> 0.954 tok/s (3.28x), top-1 and rank unchanged.
     *
     * Do NOT parallelise anything inside the head body -- the `a` loop below has
     * a real loop-carried dependency on Sh. */
    const double _p2r = p2_now();
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < vh; ++h) {
        /* v head h is fed by k/q head h % kh, NOT h / rep: the fused GDN kernel
         * uses iq1 = iv1 % neq1 (ops.cpp:11028) and ggml_repeat tiles modulo,
         * whereas h/rep gives [q0 q0 q0 q1 q1 q1 ...]. */
        const int hk = h % kh;                  /* k head feeding this v head */
        const float *khv = s->k + (size_t)hk * st;
        const float *qhv = s->q + (size_t)hk * st;
        const float *vhv = s->v + (size_t)h * st;
        const float  bh  = beta[h];
        const float  decay = expf(g[h]);
        float *Sh = S + (size_t)h * st * st;
        float *oh = s->o + (size_t)h * st;

        /* A previous audit claimed that contracting the second index instead was
         * "a different operator" whose "factor of order one on every token of
         * every GDN layer dominates the whole stack".  That is FALSE and was
         * measured to be false: the two forms are the same operator in
         * transposed layouts (see the note above the loop).  The 2.120870
         * difference in scripts/_axis_ab.py comes only from feeding the SAME
         * buffer to both loops and calling it "the same input" -- a transposed
         * buffer is a different state.  _axis_ab.py says so itself on its last
         * two lines: loop_old matches oracle(S.T) to 0.000e+00 and loop_new
         * matches oracle(S) to 0.000e+00. */
        for (int a = 0; a < st; ++a)
            for (int i = 0; i < st; ++i)
                Sh[(size_t)a * st + i] *= decay;

        for (int i = 0; i < st; ++i) {
            float sk = 0.0f;                       /* sum over the FIRST axis */
            for (int a = 0; a < st; ++a)
                sk += Sh[(size_t)a * st + i] * khv[a];

            const float d = bh * (vhv[i] - sk);    /* free index is the value axis */

            float o = 0.0f;
            for (int a = 0; a < st; ++a) {
                Sh[(size_t)a * st + i] += khv[a] * d;
                o += Sh[(size_t)a * st + i] * qhv[a];
            }
            oh[i] = o * qscale;
        }
    }
    if (p2_master()) p2_add(P2_GDN_REC, p2_now() - _p2r);

    /* ---- gated per-head RMSNorm + output projection ---- */
    snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", il);
    if (dequant_to_f32(m, nm, s->ssm_norm, (size_t)m->head_v_dim) != 0) return -1;
    for (int h = 0; h < vh; ++h) {
        float *oh = s->o + (size_t)h * st;
        rmsnorm(oh, oh, s->ssm_norm, st, m->rms_eps);
    }
    /* prism.hadamard.gdn_v_grouped: the .ssm_out. weights are the only folded
     * ones whose transform carries a head re-order (llama-model.cpp:2123-2132
     * sets perm_hd=st, perm_nk=kh, perm_rep=rep for names containing ".ssm_out.";
     * llama-graph.cpp:1561-1568 then reshapes to [hd,n_k,rep], permutes to
     * [hd,rep,n_k] and reshapes back BEFORE the sign multiply and the Hadamard).
     * On the 48-head index the net effect is  h' = rep*(h % kh) + h / kh,
     * i.e. entry h' takes head h = kh*(h' % rep) + h' / rep. */
    for (int hp = 0; hp < vh; ++hp) {
        const int h = kh * (hp % rep) + hp / rep;
        for (int d = 0; d < st; ++d)
            s->attn_out[(size_t)hp * st + d] =
                f_silu(s->z[(size_t)h * st + d]) * s->o[(size_t)h * st + d];
    }

    snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", il);
    if (mm_folded(m, s->xb2, s->attn_out, nm, m->signs_6144, value_dim,
                  n_embd) != 0)
        return -1;

    for (int i = 0; i < n_embd; ++i) s->x[i] += s->xb2[i];
    debug_tap(m, il, "block_out", s->x, n_embd);
    return 0;
}

/* Full-attention block (16 of the 64 blocks, i%4==3).
 *
 *   xb    = RMSNorm(x, attn_norm)
 *   qg    = W_q @ had(xb)        [2*n_head*head_dim]  = Q | gate
 *   k     = W_k @ had(xb), v = W_v @ had(xb)
 *   q,k   = per-head RMSNorm with attn_q_norm / attn_k_norm (over head_dim)
 *   q,k   = M-RoPE (n_rot of head_dim)
 *   GQA   = n_head/n_head_kv query heads per kv head
 *   out   = (attn . sigmoid(gate)) @ W_attn_output
 */
static int block_attn(bonsai_model *m, int il, int pos) {
    bonsai_scratch *s = &m->sc;
    const int n_embd = m->n_embd;
    const int nh = m->n_head, nkv = m->n_head_kv, hd = m->head_dim;
    /* Validate the divisor and the rope head width before using them; the open
     * path rejects these too, this keeps the block safe when driven directly. */
    if (nh <= 0 || nkv <= 0 || nh % nkv || hd <= 0 || n_embd <= 0 ||
        (int64_t)nh * hd > INT32_MAX / 2 ||
        m->n_rot <= 0 || m->n_rot > hd || m->ctx <= 0 ||
        s->scores == NULL || m->st.kv_len < 0 || m->st.kv_len >= m->ctx) {
        fprintf(stderr, "block_attn: bad geometry (nh=%d nkv=%d hd=%d n_rot=%d)\n",
                nh, nkv, hd, m->n_rot);
        return -1;
    }
    const int rep = nh / nkv;
    const int qdim = nh * hd;
    char nm[96];

    snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
    if (dequant_to_f32(m, nm, s->norm_w, (size_t)n_embd) != 0) return -1;
    rmsnorm(s->xb, s->x, s->norm_w, n_embd, m->rms_eps);

    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", il);
    if (mm_folded(m, s->qg, s->xb, nm, m->signs_5120, n_embd, 2 * qdim) != 0)
        return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", il);
    if (mm_folded(m, s->k_full, s->xb, nm, m->signs_5120, n_embd,
                  nkv * hd) != 0)
        return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", il);
    if (mm_folded(m, s->v_full, s->xb, nm, m->signs_5120, n_embd,
                  nkv * hd) != 0)
        return -1;

    /* The attn_q projection is 2*n_head*head_dim wide but is NOT laid out as
     * [all Q | all gate].  The official graph interprets it per head:
     *     Qcur_full {12288} -> view {256, 24} -> gate_reshaped {6144}
     * i.e. each head contributes {Q(head_dim), gate(head_dim)} consecutively.
     * Taking the first half as Q and the second half as the gate instead reads
     * each head's SECOND query half as its gate, which moves sigmoid(gate) from
     * ~0.03 to ~0.6 and inflates the attention output by ~3x.
     * Verified against the dump (layer 3, token 0):
     *   per-head reading first3 = [-3.2699 -4.5540 -4.1169], sigmoid sum 128.967
     *   official gate_reshaped-3 first3 = [-3.2760 -4.5583 -4.1098], sum 128.947
     *   contiguous-half reading gives sigmoid sum ~1587. */
    for (int h = 0; h < nh; ++h) {
        memcpy(s->q_full + (size_t)h * hd, s->qg + (size_t)h * 2 * hd,
               (size_t)hd * sizeof(float));
        memcpy(s->attn_gate + (size_t)h * hd,
               s->qg + (size_t)h * 2 * hd + hd, (size_t)hd * sizeof(float));
    }

    snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il);
    if (dequant_to_f32(m, nm, s->attn_qn, (size_t)hd) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il);
    if (dequant_to_f32(m, nm, s->attn_kn, (size_t)hd) != 0) return -1;
    for (int h = 0; h < nh; ++h)
        rmsnorm(s->q_full + (size_t)h * hd, s->q_full + (size_t)h * hd,
                s->attn_qn, hd, m->rms_eps);
    for (int h = 0; h < nkv; ++h)
        rmsnorm(s->k_full + (size_t)h * hd, s->k_full + (size_t)h * hd,
                s->attn_kn, hd, m->rms_eps);

    rope_tables(m->n_rot, pos, m->rope_freq_base, s->rope_cos, s->rope_sin);
    rope_apply(s->q_full, nh,  hd, m->n_rot, pos, m->rope_freq_base,
               s->rope_cos, s->rope_sin);
    rope_apply(s->k_full, nkv, hd, m->n_rot, pos, m->rope_freq_base,
               s->rope_cos, s->rope_sin);

    /* ---- append to the KV cache and attend ---- */
    const int t = m->st.kv_len;
    float *kc = m->st.kv_cache + ((size_t)il * m->ctx + t) * 2 * nkv * hd;
    float *vc = kc + (size_t)nkv * hd;
    memcpy(kc, s->k_full, (size_t)nkv * hd * sizeof(float));
    memcpy(vc, s->v_full, (size_t)nkv * hd * sizeof(float));
    const int nt = t + 1;

    const float inv_sqrt_hd = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < nh; ++h) {
        const int hkv = h / rep;
        const float *qh = s->q_full + (size_t)h * hd;
        float *sc = s->scores;
        float mx = -INFINITY;
        for (int p = 0; p < nt; ++p) {
            const float *kp = m->st.kv_cache +
                ((size_t)il * m->ctx + p) * 2 * nkv * hd + (size_t)hkv * hd;
            float acc = 0.0f;
            for (int d = 0; d < hd; ++d) acc += qh[d] * kp[d];
            acc *= inv_sqrt_hd;
            sc[p] = acc;
            if (acc > mx) mx = acc;
        }
        float sum = 0.0f;
        for (int p = 0; p < nt; ++p) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
        const float inv = 1.0f / sum;
        float *oh = s->q_full + (size_t)h * hd;      /* reuse q as output */
        for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
        for (int p = 0; p < nt; ++p) {
            const float w = sc[p] * inv;
            const float *vp = m->st.kv_cache +
                ((size_t)il * m->ctx + p) * 2 * nkv * hd + (size_t)nkv * hd +
                (size_t)hkv * hd;
            for (int d = 0; d < hd; ++d) oh[d] += w * vp[d];
        }
    }

    /* gate: the per-head second half of the fused Q projection (see above) */
    for (int i = 0; i < qdim; ++i)
        s->attn_out[i] = s->q_full[i] * f_sigmoid(s->attn_gate[i]);

    snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il);
    if (mm_folded(m, s->xb2, s->attn_out, nm, m->signs_6144, qdim,
                  n_embd) != 0)
        return -1;

    for (int i = 0; i < n_embd; ++i) s->x[i] += s->xb2[i];
    debug_tap(m, il, "block_out_attn", s->x, n_embd);
    return 0;
}

/* SwiGLU FFN, shared by both block types:
 *   xb = RMSNorm(x, post_attention_norm)
 *   h  = down( silu(gate(xb)) * up(xb) )      [all three folded] */
static int block_ffn(bonsai_model *m, int il) {
    bonsai_scratch *s = &m->sc;
    char nm[96];

    if (m->n_embd <= 0 || m->n_ff <= 0 || m->n_embd > BONSAI_FOLD_MAX ||
        m->n_ff > BONSAI_FOLD_MAX) {
        fprintf(stderr, "block_ffn: bad geometry (embd=%d ffn=%d)\n",
                m->n_embd, m->n_ff);
        return -1;
    }

    snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", il);
    if (dequant_to_f32(m, nm, s->norm_w, (size_t)m->n_embd) != 0) return -1;
    rmsnorm(s->xb, s->x, s->norm_w, m->n_embd, m->rms_eps);

    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il);
    if (mm_folded(m, s->hb, s->xb, nm, m->signs_5120, m->n_embd,
                  m->n_ff) != 0)
        return -1;
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", il);
    if (mm_folded(m, s->hb2, s->xb, nm, m->signs_5120, m->n_embd,
                  m->n_ff) != 0)
        return -1;

    for (int i = 0; i < m->n_ff; ++i) s->hb[i] = f_silu(s->hb[i]) * s->hb2[i];

    snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", il);
    if (mm_folded(m, s->xb2, s->hb, nm, m->signs_17408, m->n_ff,
                  m->n_embd) != 0)
        return -1;

    for (int i = 0; i < m->n_embd; ++i) s->x[i] += s->xb2[i];
    /* final residual of the block; for il == n_layer-1 this is what the
     * official dump prints as `l_out-63` */
    debug_tap(m, il, "block_out_ffn", s->x, m->n_embd);
    return 0;
}

/* ================================================================== */
/* embedding lookup + output head                                      */
/* ================================================================== */

/* Row lookup from the PQ2_0 embedding table, then the INVERSE transform
 * s . (H x): blockwise-1024 H first, then the width-5120 signs.
 * Verified against the official dump (model.input_embed). */
static int embed_row(bonsai_model *m, int32_t token, float *out) {
    const bonsai_tensor *t = T(m, "token_embd.weight");
    if (!t) {
        fprintf(stderr, "embed_row: missing token_embd.weight\n");
        return -1;
    }
    /* The embedding table is decoded with the PQ2_0 code map, so anything else
     * (e.g. GGML_TYPE_PTQ1_0, which has the same 34-byte block size and
     * therefore loads cleanly) must be rejected rather than decoded as garbage.
     * dequant_to_f32() rejects it; this path used not to look at t->type. */
    if (t->type != GGML_TYPE_PQ2_0) {
        fprintf(stderr, "embed_row: token_embd.weight is %s (%u), expected "
                        "PQ2_0 (%u)\n", bonsai_type_name(t->type), t->type,
                (unsigned)GGML_TYPE_PQ2_0);
        return -1;
    }
    const int n_embd = (int)t->dims[0];
    const int n_vocab = (int)t->dims[1];
    if (n_embd != m->n_embd || n_embd <= 0 || n_embd % 128 ||
        m->had_block <= 0 || n_embd % m->had_block) {
        fprintf(stderr, "embed_row: token_embd.weight width %d unusable "
                        "(model embd %d, had_block %d)\n",
                n_embd, m->n_embd, m->had_block);
        return -1;
    }
    if (token < 0 || token >= n_vocab) {
        fprintf(stderr, "embed_row: token %d out of range [0,%d)\n",
                token, n_vocab);
        return -1;
    }

    const int row_bytes = n_embd / 128 * 34;
    const uint8_t *base = bonsai_tensor_data(m->g, t) + (size_t)token * row_bytes;
    if (pq2_0_dequant_matrix(base, out, (size_t)n_embd) == 0) {
        fprintf(stderr, "embed_row: PQ2_0 decode failed for token %d\n", token);
        return -1;
    }

    if (hadamard_forward(out, NULL, n_embd, m->had_block, 0) != 0) {
        fprintf(stderr,
                "embed_row: hadamard_forward rejected n=%d block=%d\n",
                n_embd, m->had_block);
        return -1;
    }
    for (int i = 0; i < n_embd; ++i) out[i] *= m->signs_5120[i];
    debug_tap(m, -1, "embed", out, n_embd);
    return 0;
}

int bonsai_forward(bonsai_model *m, int32_t token, int32_t pos) {
    const double _p2_fwd = p2_now();
    bonsai_scratch *s = &m->sc;
    if (m->full_attn_interval <= 0) {
        fprintf(stderr, "forward: bad full_attn_interval %d\n",
                m->full_attn_interval);
        return -1;
    }
    if (pos >= m->ctx) {
        fprintf(stderr, "forward: position %d beyond context %d\n", pos, m->ctx);
        return -1;
    }
    if (m->st.kv_len != pos) {
        fprintf(stderr, "forward: kv_len %d != pos %d (state desync)\n",
                m->st.kv_len, pos);
        return -1;
    }

    if (embed_row(m, token, s->x) != 0) {
        fprintf(stderr, "forward: embedding lookup failed for token %d\n", token);
        return -1;
    }

    /* Every block can fail (missing/mismatched/unsupported weight, bad
     * geometry). Previously those failures were swallowed: mm_folded() returned
     * early leaving `y` stale, so the logits kept the previous token's values
     * while bonsai_forward() still reported success. Now the first failure
     * aborts the token with a non-zero return. */
    for (int il = 0; il < m->n_layer; ++il) {
        const int is_full_attn = ((il + 1) % m->full_attn_interval == 0);
        const double _p2b = p2_now();
        const int rc = is_full_attn ? block_attn(m, il, pos) : block_gdn(m, il);
        if (p2_master()) p2_add(is_full_attn ? P2_ATTN : P2_GDN, p2_now() - _p2b);
        if (rc != 0) {
            fprintf(stderr, "forward: block %d failed\n", il);
            return -1;
        }
        const double _p2f = p2_now();
        if (block_ffn(m, il) != 0) {
            fprintf(stderr, "forward: block %d ffn failed\n", il);
            return -1;
        }
        if (p2_master()) p2_add(P2_FFN, p2_now() - _p2f);
    }
    m->st.kv_len = pos + 1;

    /* final norm + output head (folded with the width-5120 signs) */
    if (dequant_to_f32(m, "output_norm.weight", s->norm_w,
                       (size_t)m->n_embd) != 0)
        return -1;
    rmsnorm(s->xb, s->x, s->norm_w, m->n_embd, m->rms_eps);
    if (mm_folded(m, s->logits, s->xb, "output.weight", m->signs_5120,
                  m->n_embd, m->n_vocab) != 0)
        return -1;
    if (p2_master()) p2_add(P2_FWD, p2_now() - _p2_fwd);
#ifdef BONSAI_PROF2
    /* Dumped from HERE, not from main_k1om.c: the counters are static-in-header,
     * so each translation unit has its own copy and only this one has the data.
     * Only on the last call, so a 9-token run prints one budget. */
    if (pos == 8) p2_dump();   /* last token of the 5+3 test run */
#endif
    return 0;
}

int bonsai_generate(bonsai_model *m, const int32_t *prompt, int32_t n_prompt,
                    int32_t *out, int32_t n_out) {
    int32_t pos = 0;
    for (int32_t i = 0; i < n_prompt; ++i) {
        if (bonsai_forward(m, prompt[i], pos++) != 0) return -1;
    }
    int32_t produced = 0;
    for (int32_t i = 0; i < n_out; ++i) {
        int32_t best = 0;
        float bv = m->sc.logits[0];
        for (int32_t v = 1; v < m->n_vocab; ++v) {
            if (m->sc.logits[v] > bv) { bv = m->sc.logits[v]; best = v; }
        }
        out[produced++] = best;
        if (i + 1 < n_out) {
            if (pos >= m->ctx) break;
            if (bonsai_forward(m, best, pos++) != 0) break;
        }
    }
    return produced;
}
