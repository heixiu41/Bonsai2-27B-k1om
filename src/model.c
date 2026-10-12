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
 * instead, so the macro is only needed for the host build. */
#ifndef __MIC__
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200112L
#  endif
#endif

#include "model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Widest folded input (ffn_down). sc.fold_x holds one of these and mm_stage()
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
#ifdef __MIC__
    /* The serial double-precision accumulation chain costs ~6 cyc/element
     * (129 calls/token x 5120 elements ~ 3.2 ms/token, static analysis of
     * the pragma census). On the card, sum 8-wide in double with four
     * independent chains. IMCI has NO _mm512_cvtps_pd, so each 16-float
     * vector is converted as low half + permuted high half (cvtpslo_pd).
     * This REASSOCIATES the double adds (~1e-7 drift on the card only; the
     * host keeps the scalar chain below, so the oracle regression is
     * untouched; the card gate is top-1/rank). */
    {
        /* lane j of the result takes source lane perm[j]: move floats 8..15
         * into the low half for the second cvtpslo. */
        const __m512i HI = _mm512_set_epi32(7,6,5,4,3,2,1,0,
                                            15,14,13,12,11,10,9,8);
        __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
        __m512d a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
        int i = 0;
        /* scratch buffers are calloc'd (16 B-aligned at best) and IMCI has
         * NO unaligned 512-bit load: walk scalar to the 64 B boundary first.
         * All per-head call sites offset by multiples of 512 B, so the base
         * determines alignment. */
        while (i < n && ((uintptr_t)(x + i) & 63u)) {
            ss += (double)x[i] * (double)x[i];
            ++i;
        }
        for (; i + 32 <= n; i += 32) {
            const __m512 v0 = _mm512_load_ps(x + i);
            const __m512d lo0 = _mm512_cvtpslo_pd(v0);
            const __m512d hi0 = _mm512_cvtpslo_pd(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(HI, _mm512_castps_si512(v0))));
            const __m512 v1 = _mm512_load_ps(x + i + 16);
            const __m512d lo1 = _mm512_cvtpslo_pd(v1);
            const __m512d hi1 = _mm512_cvtpslo_pd(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(HI, _mm512_castps_si512(v1))));
            a0 = _mm512_fmadd_pd(lo0, lo0, a0);
            a1 = _mm512_fmadd_pd(hi0, hi0, a1);
            a2 = _mm512_fmadd_pd(lo1, lo1, a2);
            a3 = _mm512_fmadd_pd(hi1, hi1, a3);
        }
        for (; i + 16 <= n; i += 16) {
            const __m512 v = _mm512_load_ps(x + i);
            const __m512d lo = _mm512_cvtpslo_pd(v);
            const __m512d hi = _mm512_cvtpslo_pd(_mm512_castsi512_ps(
                _mm512_permutevar_epi32(HI, _mm512_castps_si512(v))));
            a0 = _mm512_fmadd_pd(lo, lo, a0);
            a1 = _mm512_fmadd_pd(hi, hi, a1);
        }
        for (; i < n; ++i) ss += (double)x[i] * (double)x[i];
        a0 = _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3));
        ss += _mm512_reduce_add_pd(a0);
    }
#else
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
#endif
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

/* mm_folded() is split into a STAGING half (mm_stage) and a WEIGHT half
 * (mm_run), because several call groups share the SAME input buffer and sign
 * vector and were re-running the identical staging computation per member:
 *   block_gdn : attn_qkv + attn_gate     (s->xb, signs_5120)  1 redundant
 *   block_attn: attn_q + attn_k + attn_v (s->xb, signs_5120)  2 redundant
 *   block_ffn : ffn_gate + ffn_up        (s->xb, signs_5120)  1 redundant
 * i.e. 143 of the 401 per-token Hadamard transforms were pure recompute.
 * Grouped sites stage once and call mm_run() per member; single-use sites
 * keep the mm_folded() wrapper. Bit-identical by construction: the same
 * staging arithmetic, performed once instead of twice/thrice. The "signed"
 * and "hx" debug taps now fire once per group instead of once per member --
 * the same content is tapped. */

/* Stage `x` into the 64-byte-aligned sc.fold_x (memcpy, optional sign
 * multiply, optional blockwise Hadamard). Returns -1 on a bad input width.
 *
 * EVERY input is staged through fold_x before it reaches the IMCI kernel:
 * pq2_0_matvec_knc() faults on a merely 16-byte-aligned x (calloc default),
 * and only the staging code knows whether the caller's buffer qualifies.
 * This used to be true only for signs != NULL, which left mm_plain() -- and
 * any other signs==NULL caller -- as an alignment landmine that could only
 * fire on the card. */
static int mm_stage(bonsai_model *m, const float *x, const float *signs,
                    int n) {
    if (n <= 0 || n > (int)BONSAI_FOLD_MAX) {
        fprintf(stderr, "mm_stage: input width %d out of range (cap %d)\n",
                n, (int)BONSAI_FOLD_MAX);
        return -1;
    }
    if (signs) {
        if (m->had_block <= 0 || n % m->had_block) {
            /* hadamard_forward() steps base += block without checking n % block
             * and would write past the end of the input for a non-multiple. */
            fprintf(stderr, "mm_stage: input width %d is not a multiple of "
                            "had_block %d\n", n, m->had_block);
            return -1;
        }
        memcpy(m->sc.fold_x, x, (size_t)n * sizeof(float));
#ifdef __MIC__
        /* fold_x and the signs arrays are both 64 B-aligned (aligned_f32);
         * IMCI has no unaligned 512-bit load. Scalar tail covers n%16. */
        {
            int i = 0;
            for (; i + 16 <= n; i += 16)
                _mm512_store_ps(m->sc.fold_x + i,
                    _mm512_mul_ps(_mm512_load_ps(m->sc.fold_x + i),
                                  _mm512_load_ps(signs + i)));
            for (; i < n; ++i) m->sc.fold_x[i] *= signs[i];
        }
#else
        for (int i = 0; i < n; ++i) m->sc.fold_x[i] *= signs[i];
#endif
        if (m->debug_want && strcmp(m->debug_want, "signed") == 0 && !m->debug_len) {
            int c = n < m->debug_cap ? n : m->debug_cap;
            memcpy(m->debug, m->sc.fold_x, (size_t)c * sizeof(float));
            m->debug_len = c;
        }
        /* hadamard_forward now reports misuse (block<=0 or block !| n) instead
         * of silently writing past the buffer. bonsai_model_open validates the
         * widths up front, so a failure here means the two disagree. */
        if (hadamard_forward(m->sc.fold_x, NULL, n, m->had_block, 0) != 0) {
            fprintf(stderr,
                    "mm_stage: hadamard_forward rejected n=%d block=%d\n",
                    n, m->had_block);
            return -1;
        }
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
    /* Build the permuted activation ONCE per stage (parallel over 128-float
     * blocks, bit-identical to the serial builder): every mm_run() against
     * this staged input then skips its own rebuild. */
    pq2_build_xp_mt(m->sc.fold_x, m->sc.fold_xp, n);
#endif
    return 0;
}

/* Run the PQ2_0 matvec for `wname` against the already-staged sc.fold_x.
 * Returns 0 on success; -1 when the weight is missing, is not PQ2_0, or does
 * not match the caller's geometry. On failure `y` is left untouched, so the
 * caller MUST abort the token: returning silently used to leave `y` holding
 * the previous token's values while bonsai_forward() still reported success.
 *
 * The tensor type is checked explicitly: gguf.h gives GGML_TYPE_PTQ1_0 (143)
 * the same 34-byte block size as PQ2_0, so such a tensor loads cleanly and
 * would otherwise be decoded with the PQ2_0 code map, producing garbage with
 * no error at all (dequant_to_f32() rejects it, but the mm path used not to
 * look). */

#ifdef __MIC__
/* Lazy per-tensor drow tables (proven bit-identical by src/probes/pq2drow.c,
 * 1.142x at 244t, ~18.2 ms/token). LAZY, not load-time: bonsai_model_open()
 * runs while the 7.2 GB tmpfs model file still exists (main_k1om.c unlinks
 * only after open returns), so the load peak is tmpfs 7.2 + anon 7.7 GB and
 * building +840 MB of tables inside open OOM-killed the card (dmesg
 * confirmed). Built on first mm_run per tensor instead: by then the file is
 * unlinked and ~7 GB is free. mm_run only ever runs on the master thread
 * (the kernels open their own omp regions), so the write-back is race-free.
 * BONSAI_NO_DROW disables; allocation failure is per-tensor and permanent
 * (PQ2_DROWS_FAILED), falling back to the per-call builder. */
#define PQ2_DROWS_FAILED ((float *)(uintptr_t)1)
static int mm_drows_disabled = -1;      /* -1 unknown, 0 enabled, 1 disabled */
static int mm_pack_disabled = -1;       /* BONSAI_NO_PACK: A/B switch */
static uint64_t mm_drows_bytes = 0;
static int mm_drows_count = 0;

static const float *mm_drows_for(bonsai_model *m, const bonsai_tensor *tc,
                                 int n, int d) {
    if (tc->drows == PQ2_DROWS_FAILED) return NULL;
    if (tc->drows) return tc->drows;
    if (mm_drows_disabled < 0)
        mm_drows_disabled = getenv("BONSAI_NO_DROW") != NULL;
    if (mm_drows_disabled) return NULL;
    bonsai_tensor *t = (bonsai_tensor *)tc;   /* lazy cache write-back */
    const int nblk     = n / QK_PQ2_0;
    const int nblk_pad = (nblk + 15) & ~15;   /* 64 B row-base alignment */
    const size_t rb    = (size_t)nblk * PQ2_0_BLOCK_BYTES;
    float *dr = aligned_f32((size_t)nblk_pad * (size_t)d, 64);
    if (!dr) {
        fprintf(stderr, "model: drow allocation failed for %s (%d x %d) "
                "-- per-call builder\n", t->name, nblk_pad, d);
        t->drows = PQ2_DROWS_FAILED;
        return NULL;
    }
    pq2_build_drows_mt(bonsai_tensor_data(m->g, t), rb, nblk, d, dr);
    t->drows = dr;
    /* Codes-only repack, immediately after (and never before) the drows read
     * the fp16 scales: 34 -> 32 bytes per block, in place, no new allocation.
     * Every later mm_run of this tensor takes the packed kernel.  The cast is
     * the deliberate exception to bonsai_tensor_data()'s const: this rewrites
     * the loader's own copy of the weights, once, before any other reader. */
    /* Codes-only repack (34 -> 32 bytes per block, in place, no allocation).
     * OPT-IN (BONSAI_PACK=1), because the end-to-end A/B could not show a win:
     * interleaved same-binary 64-token decode gave 3.770 tok/s at 34 bytes vs
     * 3.747 at 32 (median of 3 each, -0.62%).  The kernel itself IS faster --
     * the probe measures +7.8%..+9.2% at 1 thread, bit-identical on all five
     * shapes -- because the saving is instruction count, and at 244 threads
     * this matvec is memory-LATENCY bound, so 2 of ~30 instructions do not
     * matter.  It also cannot shrink RSS: the holes are scattered through the
     * same allocation.  Kept because the mechanism is proven and the regime
     * where it pays (few threads) is exactly what a future single-threaded or
     * issue-bound variant would use. */
    if (mm_pack_disabled < 0)
        mm_pack_disabled = (getenv("BONSAI_PACK") == NULL);
    if (!mm_pack_disabled) {
        pq2_repack_rows_inplace((uint8_t *)bonsai_tensor_data(m->g, t), nblk, d);
        t->packed = 1;
    }
    mm_drows_bytes += (uint64_t)nblk_pad * (uint64_t)d * sizeof(float);
    if ((++mm_drows_count & 127) == 0)
        printf("drows: %d tensors, %.1f MB (lazy)\n", mm_drows_count,
               (double)mm_drows_bytes / 1048576.0);
    return dr;
}
#endif

static int mm_run(bonsai_model *m, float *y, const char *wname, int n,
                  int d_cap) {
    const bonsai_tensor *t = T(m, wname);
    if (!t) {
        fprintf(stderr, "mm_run: missing weight %s\n", wname);
        return -1;
    }
    if (t->type != GGML_TYPE_PQ2_0) {
        fprintf(stderr, "mm_run: %s is %s (%u), expected PQ2_0 (%u)\n",
                wname, bonsai_type_name(t->type), t->type,
                (unsigned)GGML_TYPE_PQ2_0);
        return -1;
    }
    if (n <= 0 || t->n_dims < 2 || n > (int)BONSAI_FOLD_MAX ||
        t->dims[0] != (uint64_t)n || t->dims[1] > (uint64_t)d_cap) {
        fprintf(stderr, "mm_run: %s shape mismatch (dims {%llu,%llu}, "
                        "caller n=%d cap=%d)\n", wname,
                (unsigned long long)t->dims[0],
                (unsigned long long)t->dims[1], n, d_cap);
        return -1;
    }
    const uint8_t *w = bonsai_tensor_data(m->g, t);
    const int d = (int)t->dims[1];      /* ggml ne1 = output dim */

#ifdef __MIC__
    /* Rescale variant of the F3 block loop: one fp32 rounding per 128-weight
     * block instead of one per element, i.e. a ~1e-7-class numerical drift
     * (see pq2_0.h:pq2_row_block_sum).  OFF by default: the A/B could not
     * establish a win (it measured +2.7% and -2.1% in two runs of the same
     * comparison, inside +-6% cross-process drift), so the bit-exact F3 order
     * stays the default and BONSAI_RESCALE=1 opts in. */
    const int rescale = pq2_rescale_default();
    /* Precomputed drow table (built lazily on first use; bit-identical to
     * the per-call builder, proven in src/probes/pq2drow.c). */
    const float *dr = mm_drows_for(m, t, n, d);
    if (dr && t->packed)
        /* codes-only rows: 32-byte stride, one aligned upconverting load per
         * 16 code bytes instead of the two-instruction unpack pair */
        pq2_0_matvec_knc_xp_packed(y, m->sc.fold_xp, w, n, d, dr, rescale);
    else if (dr)
        pq2_0_matvec_knc_xp_dpre(y, m->sc.fold_xp, w, n, d, dr, rescale);
    else
        pq2_0_matvec_knc_xp(y, m->sc.fold_xp, w, n, d, rescale);
#else
    pq2_0_matvec_ref(y, m->sc.fold_x, w, n, d);
#endif
    return 0;
}

/* Single-use wrapper: stage `x`, then run `wname`. Grouped call sites use
 * mm_stage() once + mm_run() per member instead (see the note above). */
static int mm_folded(bonsai_model *m, float *y, const float *x,
                     const char *wname, const float *signs, int n, int d_cap) {
    if (mm_stage(m, x, signs, n) != 0) return -1;
    return mm_run(m, y, wname, n, d_cap);
}

/* (matvec_beta_alpha was inlined into block_gdn's single parallel region;
 * see the region's first `omp for` block.) */

/* ================================================================== */
/* M-RoPE over the first n_rot dims, NeoX pairing (i, i+n_rot/2)        */
/*                                                                     */
/* theta_i = pos * freq_base^(-2i/n_rot), i in [0, n_rot/2)            */
/* Verified against ggml-cpu/ops.cpp rotate_pairs() with               */
/* n_offset = n_dims/2 and scale = 2.                                  */
/* ================================================================== */
/* Tables from the position-independent frequency cache (m->rope_freq,
 * filled once in bonsai_model_open): theta is the same expression
 * pos * freq[i] as the original pow()-per-call version, so the output is
 * bit-identical; the tables now cost n_rot/2 cos+sin per TOKEN (computed
 * once in bonsai_forward) instead of pow+cos+sin per attn LAYER. */
static void rope_tables_cached(const bonsai_model *m, int pos,
                               float *cos_t, float *sin_t) {
    const int half = m->n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const double theta = (double)pos * m->rope_freq[i];
        cos_t[i] = (float)cos(theta);
        sin_t[i] = (float)sin(theta);
    }
}

/* per-head body of rope_apply, for the parallel region in block_attn */
static inline void rope_apply_head(float *v, int half,
                                   const float *cos_t, const float *sin_t) {
    for (int i = 0; i < half; ++i) {
        const float c = cos_t[i], s = sin_t[i];
        const float a = v[i];
        const float b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
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

/* Layer-slot maps for the weight cache (model.h: bonsai_wcache): GDN arrays
 * are indexed by g = il - (il+1)/interval and attention arrays by
 * a = (il+1)/interval - 1, so no cache memory is spent on the other block
 * type. For interval 4: layers 0,1,2 -> g 0,1,2; layer 3 -> a 0; layer 4 ->
 * g 3; ... */
static int gdn_slot(const bonsai_model *m, int il) {
    return il - (il + 1) / m->full_attn_interval;
}
static int attn_slot(const bonsai_model *m, int il) {
    return (il + 1) / m->full_attn_interval - 1;
}

/* Fill the token-invariant weight cache through the same dequant_to_f32()
 * the per-token path used to call on every token, so every cached float is
 * bit-identical to the value the old code produced. Any failure must abort
 * the open: the block functions no longer consult the tensor file for these
 * tables and would otherwise read an incomplete cache silently. */
static int wcache_fill(bonsai_model *m) {
    bonsai_wcache *wc = &m->wc;
    const int n_embd = m->n_embd, vh = m->ssm_dt_rank, hd = m->head_dim;
    char nm[96];
    for (int il = 0; il < m->n_layer; ++il) {
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
        if (dequant_to_f32(m, nm, wc->attn_norm + (size_t)il * n_embd,
                           (size_t)n_embd) != 0) return -1;
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", il);
        if (dequant_to_f32(m, nm, wc->post_norm + (size_t)il * n_embd,
                           (size_t)n_embd) != 0) return -1;
        if ((il + 1) % m->full_attn_interval != 0) {          /* GDN block */
            const size_t g = (size_t)gdn_slot(m, il);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", il);
            if (dequant_to_f32(m, nm, wc->ssm_beta + g * n_embd * vh,
                               (size_t)n_embd * vh) != 0) return -1;
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", il);
            if (dequant_to_f32(m, nm, wc->ssm_alpha + g * n_embd * vh,
                               (size_t)n_embd * vh) != 0) return -1;
            snprintf(nm, sizeof(nm), "blk.%d.ssm_a", il);
            if (dequant_to_f32(m, nm, wc->ssm_a + g * vh, (size_t)vh) != 0)
                return -1;
            snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", il);
            if (dequant_to_f32(m, nm, wc->ssm_dt + g * vh, (size_t)vh) != 0)
                return -1;
            snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", il);
            if (dequant_to_f32(m, nm,
                    wc->ssm_conv1d + g * m->ssm_d_conv * m->conv_dim,
                    (size_t)m->ssm_d_conv * m->conv_dim) != 0) return -1;
            snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", il);
            if (dequant_to_f32(m, nm, wc->ssm_norm + g * m->head_v_dim,
                               (size_t)m->head_v_dim) != 0) return -1;
        } else {                                       /* full attention */
            const size_t a = (size_t)attn_slot(m, il);
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il);
            if (dequant_to_f32(m, nm, wc->attn_qn + a * hd, (size_t)hd) != 0)
                return -1;
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il);
            if (dequant_to_f32(m, nm, wc->attn_kn + a * hd, (size_t)hd) != 0)
                return -1;
        }
    }
    if (dequant_to_f32(m, "output_norm.weight", wc->output_norm,
                       (size_t)n_embd) != 0) return -1;
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

    /* How many blocks are full attention (the rest are GDN / linear). The KV
     * cache is sized and indexed by this, not by n_layer: only these blocks
     * ever touch it, and n_layer would over-allocate 4x here (64 vs 16). */
    m->n_attn = 0;
    for (int il = 0; il < m->n_layer; ++il)
        if ((il + 1) % m->full_attn_interval == 0) ++m->n_attn;
    if (m->n_attn <= 0) {
        fprintf(stderr, "model: no full-attention block in %d layers "
                        "(interval %d)\n", m->n_layer, m->full_attn_interval);
        bonsai_model_close(m);
        return NULL;
    }
    /* context: default to something that fits comfortably (the reference caps
     * at 2048 on this hardware class). */
    if (ctx <= 0) ctx = 2048;
    m->ctx = ctx;

    /* ---- Hadamard sign vectors ----
     * aligned_f32 (64 B): the k1om staging path multiplies them into fold_x
     * with aligned 512-bit vector ops -- IMCI has NO unaligned load, so a
     * plain malloc() would be an alignment landmine there. */
    if (m->had_block > 0 && g->had_sign_width_count > 0) {
        m->signs_5120  = aligned_f32(5120, 64);
        m->signs_6144  = aligned_f32(6144, 64);
        m->signs_17408 = aligned_f32(17408, 64);
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
    m->st.kv_cache  = (float *)calloc((size_t)m->n_attn * (size_t)ctx * 2 *
                                      m->n_head_kv * m->head_dim,
                                      sizeof(float));
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
    s->fold_xp = aligned_f32(BONSAI_FOLD_MAX, 64);

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
        !s->fold_xp) {
        fprintf(stderr, "model: allocation failed\n");
        bonsai_model_close(m);
        return NULL;
    }

    /* ---- token-invariant weight cache (model.h: bonsai_wcache) ----
     * Sized like the old per-token scratch tables but holding ALL layers at
     * once (~110 MB): the per-token dequant_to_f32() calls in the block
     * functions used to redo this work every token (~85 ms/token on the card
     * for the ssm_beta/ssm_alpha pair alone). */
    int n_attn = m->n_attn;                 /* computed once above */
    const int n_gdn = m->n_layer - n_attn;

    bonsai_wcache *wc = &m->wc;
    wc->attn_norm   = (float *)calloc((size_t)m->n_layer * n_embd,
                                      sizeof(float));
    wc->post_norm   = (float *)calloc((size_t)m->n_layer * n_embd,
                                      sizeof(float));
    wc->output_norm = (float *)calloc(n_embd, sizeof(float));
    wc->ssm_beta    = (float *)calloc((size_t)n_gdn * n_embd * vh,
                                      sizeof(float));
    wc->ssm_alpha   = (float *)calloc((size_t)n_gdn * n_embd * vh,
                                      sizeof(float));
    wc->ssm_a       = (float *)calloc((size_t)n_gdn * vh, sizeof(float));
    wc->ssm_dt      = (float *)calloc((size_t)n_gdn * vh, sizeof(float));
    wc->ssm_norm    = (float *)calloc((size_t)n_gdn * m->head_v_dim,
                                      sizeof(float));
    wc->ssm_conv1d  = (float *)calloc((size_t)n_gdn * m->ssm_d_conv * conv_dim,
                                      sizeof(float));
    wc->attn_qn     = (float *)calloc((size_t)n_attn * head_dim, sizeof(float));
    wc->attn_kn     = (float *)calloc((size_t)n_attn * head_dim, sizeof(float));
    if (!wc->attn_norm || !wc->post_norm || !wc->output_norm ||
        !wc->ssm_beta || !wc->ssm_alpha || !wc->ssm_a || !wc->ssm_dt ||
        !wc->ssm_norm || !wc->ssm_conv1d || !wc->attn_qn || !wc->attn_kn) {
        fprintf(stderr, "model: weight-cache allocation failed\n");
        bonsai_model_close(m);
        return NULL;
    }
    if (wcache_fill(m) != 0) {
        fprintf(stderr, "model: weight-cache dequantisation failed\n");
        bonsai_model_close(m);
        return NULL;
    }

    /* RoPE frequency cache: freq_base^(-2i/n_rot) is position-independent.
     * Computed once here; per token only cos/sin(pos*freq[i]) run (once,
     * not per attention layer). Same expression as rope_tables() ->
     * bit-identical. */
    m->rope_freq = (double *)malloc((size_t)(m->n_rot / 2) * sizeof(double));
    if (!m->rope_freq) { bonsai_model_close(m); return NULL; }
    for (int i = 0; i < m->n_rot / 2; ++i)
        m->rope_freq[i] = pow(m->rope_freq_base, -2.0 * i / m->n_rot);

#ifdef __MIC__
    /* Absorb the one-time OpenMP runtime warmup (~350 ms, measured as token
     * 0 costing 0.72 s vs 0.37 s steady-state) here at load, so the first
     * real token is not charged for it. */
    #pragma omp parallel
    { }
#endif
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
    aligned_f32_free(s->fold_xp);
    free(m->rope_freq);
    free(m->wc.attn_norm); free(m->wc.post_norm); free(m->wc.output_norm);
    free(m->wc.ssm_beta); free(m->wc.ssm_alpha); free(m->wc.ssm_a);
    free(m->wc.ssm_dt); free(m->wc.ssm_norm); free(m->wc.ssm_conv1d);
    free(m->wc.attn_qn); free(m->wc.attn_kn);
    aligned_f32_free(m->signs_5120); aligned_f32_free(m->signs_6144);
    aligned_f32_free(m->signs_17408);
    /* per-tensor drow tables (k1om only; NULL elsewhere, aligned_f32_free
     * tolerates NULL). Must run before bonsai_gguf_free drops the table. */
    if (m->g) {
        for (uint64_t i = 0; i < m->g->tensor_len; ++i)
            aligned_f32_free(m->g->tensors[i].drows);
    }
    bonsai_gguf_free(m->g);    free(m);
}

uint64_t bonsai_model_runtime_bytes(const bonsai_model *m) {
    uint64_t b = 0;
    b += (uint64_t)m->n_layer * m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state * 4;
    b += (uint64_t)m->n_layer * (m->ssm_d_conv - 1) * m->conv_dim * 4;
    b += (uint64_t)m->n_attn * m->ctx * 2 * m->n_head_kv * m->head_dim * 4;
    b += (uint64_t)m->n_embd * 3 * 4;
    b += (uint64_t)m->conv_dim * 2 * 4;
    b += (uint64_t)m->value_dim * 2 * 4;
    b += (uint64_t)m->ssm_dt_rank * m->ssm_d_state * 4 * 4;
    b += (uint64_t)m->n_ff * 2 * 4;
    b += (uint64_t)m->n_vocab * 4;
    b += (uint64_t)2 * m->n_head * m->head_dim * 4;
    b += (uint64_t)m->n_head * m->ctx * 4;
    b += 17408u * 4;      /* fold_x  */
    b += 17408u * 4;      /* fold_xp */
    b += 5120u * 4 + 6144u * 4 + 17408u * 4;
    /* token-invariant weight cache (model.h: bonsai_wcache) */
    uint64_t n_attn = (uint64_t)m->n_attn;
    const uint64_t n_gdn = (uint64_t)m->n_layer - n_attn;
    b += (uint64_t)m->n_layer * m->n_embd * 2 * 4;       /* attn/post norms */
    b += (uint64_t)m->n_embd * 4;                        /* output_norm */
    b += n_gdn * m->n_embd * m->ssm_dt_rank * 2 * 4;     /* ssm_beta/alpha */
    b += n_gdn * m->ssm_dt_rank * 2 * 4;                 /* ssm_a/ssm_dt */
    b += n_gdn * m->head_v_dim * 4;                      /* ssm_norm */
    b += n_gdn * m->ssm_d_conv * m->conv_dim * 4;        /* ssm_conv1d */
    b += n_attn * m->head_dim * 2 * 4;                   /* attn q/k norms */
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
    {
        /* upper bound for the lazily-built per-tensor drow tables (k1om
         * only; built on first mm_run, after the tmpfs model is unlinked) */
        uint64_t drb = 0;
        for (uint64_t i = 0; i < m->g->tensor_len; ++i) {
            const bonsai_tensor *t = &m->g->tensors[i];
            if (t->type == GGML_TYPE_PQ2_0 && t->n_dims >= 2 &&
                t->dims[0] % QK_PQ2_0 == 0)
                drb += t->dims[1] *
                       (((t->dims[0] / QK_PQ2_0) + 15) & ~(uint64_t)15) * 4;
        }
        printf("  drows (lazy, worst case)   : %.3f GB\n",
               (double)drb / 1e9);
    }
    printf("  runtime state + scratch  : %.3f GB\n",
           (double)bonsai_model_runtime_bytes(m) / 1e9);
}

void bonsai_reset(bonsai_model *m) {
    const size_t L = (size_t)m->n_layer;
    memset(m->st.ssm_state, 0, L * m->ssm_dt_rank * m->ssm_d_state *
                               m->ssm_d_state * sizeof(float));
    memset(m->st.conv_hist, 0, L * (m->ssm_d_conv - 1) * m->conv_dim *
                               sizeof(float));
    memset(m->st.kv_cache, 0, (size_t)m->n_attn * (size_t)m->ctx * 2 *
                              m->n_head_kv * m->head_dim * sizeof(float));
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

    /* All small constant weights now come from the load-time cache
     * (model.h: bonsai_wcache); the per-token dequant_to_f32() calls that used
     * to be here redid ~85 ms/token of identical BF16->F32 work on the card. */
    const bonsai_wcache *wc = &m->wc;
    const size_t gslot = (size_t)gdn_slot(m, il);

    rmsnorm(s->xb, s->x, wc->attn_norm + (size_t)il * n_embd, n_embd,
            m->rms_eps);
    if (!m->debug_len) debug_tap(m, il, "attn_norm", s->xb, n_embd);

    /* attn_qkv and attn_gate share the SAME input (s->xb) and sign vector:
     * stage the sign+Hadamard transform once and run both matvecs against it
     * (one of the two Hadamards of this group was pure recompute). */
    if (mm_stage(m, s->xb, m->signs_5120, n_embd) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", il);
    if (mm_run(m, s->qkv, nm, n_embd, conv_dim) != 0) return -1;
    debug_tap(m, il, "qkv", s->qkv, conv_dim);

    snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", il);
    if (mm_run(m, s->z, nm, n_embd, value_dim) != 0) return -1;

    /* beta|alpha weights: [n_embd][vh] row-major tables (BF16 source, cached
     * as fp32); beta = rows [0, vh), alpha = rows [vh, 2*vh) of beta_out. */
    const float *w_beta  = wc->ssm_beta  + gslot * n_embd * vh;
    const float *w_alpha = wc->ssm_alpha + gslot * n_embd * vh;
    const float *ssm_a  = wc->ssm_a  + gslot * vh;
    const float *ssm_dt = wc->ssm_dt + gslot * vh;
    float *beta = s->beta_out;
    float *g    = s->beta_out + vh;      /* reuse: alpha -> g */
    /* conv weight layout, verified against the official dump: the tensor
     * claims dims={4, 10240} but the data is stored CHANNEL-MAJOR with the
     * kernel tap fastest, i.e. flat[c*conv_k + k]. */
    const float *w_conv = wc->ssm_conv1d + gslot * ck * conv_dim;
    float *hist = m->st.conv_hist + (size_t)il * (ck - 1) * conv_dim;
    const float *ssm_norm = wc->ssm_norm + gslot * m->head_v_dim;
    const float qscale = 1.0f / sqrtf((float)st);
    float *S = m->st.ssm_state + (size_t)il * vh * st * st;

    /* ONE parallel region for the whole GDN glue stage. The old code paid a
     * ~34 us fork/join for EACH of six small stages per layer (352 glue
     * regions per token ~ 12 ms, per the static region census) plus serial
     * beta/g and v-copy loops. The implicit barriers between `omp for`
     * blocks preserve the original data dependencies (beta/alpha rows ->
     * transcendentals; conv -> hist shift; conv -> l2norm/v-copy; l2norm +
     * v + beta/g -> recurrence; recurrence -> rmsnorm -> silu), and every
     * per-element/per-head arithmetic order is untouched, so host and card
     * results are unchanged. Without OpenMP the body runs top-to-bottom in
     * the old order. */
    #pragma omp parallel
    {
        /* beta/alpha projection rows over the cached weights (this is the
         * old matvec_beta_alpha body, inlined so it joins this region) */
        #pragma omp for schedule(static)
        for (int o = 0; o < 2 * vh; ++o) {
            const float *row = (o < vh ? w_beta : w_alpha) +
                               (size_t)(o < vh ? o : o - vh) * n_embd;
            float acc = 0.0f;
            for (int i = 0; i < n_embd; ++i) acc += row[i] * s->xb[i];
            s->beta_out[o] = acc;
        }

        #pragma omp for schedule(static)
        for (int h = 0; h < vh; ++h) {
            beta[h] = f_sigmoid(beta[h]);
            g[h] = f_softplus(g[h] + ssm_dt[h]) * ssm_a[h];
        }

        /* ---- causal depthwise conv over the conv_dim channels, kernel 4 --
         * Channels are independent (each writes only conv_out[c] and reads
         * hist[c], qkv[c], w[c*ck..]). The history SHIFT below stays single:
         * its memcpy regions overlap deliberately. ck >= 2 is guaranteed by
         * the geometry checks in bonsai_model_open(). */
        #pragma omp for schedule(static)
        for (int c = 0; c < conv_dim; ++c) {
            const float *wch = w_conv + (size_t)c * ck;
            float acc = 0.0f;
            for (int k = 0; k < ck - 1; ++k)          /* oldest .. second-newest */
                acc += wch[k] * hist[(size_t)k * conv_dim + c];
            acc += wch[ck - 1] * s->qkv[c];           /* the current token */
            s->conv_out[c] = f_silu(acc);
        }
        #pragma omp single
        {
            for (int k = 0; k < ck - 2; ++k)
                memcpy(hist + (size_t)k * conv_dim, hist + (size_t)(k + 1) * conv_dim,
                       (size_t)conv_dim * sizeof(float));
            memcpy(hist + (size_t)(ck - 2) * conv_dim, s->qkv,
                   (size_t)conv_dim * sizeof(float));
            debug_tap(m, il, "conv", s->conv_out, conv_dim);
        }

        /* ---- L2 normalisation: ONE SCALE PER HEAD (see the long comment in
         * git history: the official op is 32 independent norms of 128, NOT
         * one shared scale per block) ---- */
        #pragma omp for schedule(static)
        for (int h = 0; h < kh; ++h) {
            l2norm(s->q + (size_t)h * st, s->conv_out + (size_t)h * st, st);
            l2norm(s->k + (size_t)h * st,
                   s->conv_out + key_dim + (size_t)h * st, st);
        }

        /* v is not normalised */
        #pragma omp for schedule(static)
        for (int i = 0; i < value_dim; ++i)
            s->v[i] = s->conv_out[2 * key_dim + i];

        /* ---- DeltaNet recurrence, per value head ----
         * Physical state layout: element (a, i) lives at Sh[i*st + a]
         * (transposed), so inner loops walk consecutive floats (the old
         * layout compiled to vgatherdps retry loops). The 48 value heads are
         * provably independent: head h touches only its own S slice, reads
         * k/q/v/beta/g read-only, writes only its own o slice.
         * v head h is fed by k/q head h % kh (modulo tiling, NOT h / rep).
         * Do NOT parallelise inside the head body -- the inner loops carry
         * the recurrence (the rank-1 update feeds the o accumulation). */
        #pragma omp for schedule(static)
        for (int h = 0; h < vh; ++h) {
            const int hk = h % kh;                  /* k head feeding this v head */
            const float *khv = s->k + (size_t)hk * st;
            const float *qhv = s->q + (size_t)hk * st;
            const float *vhv = s->v + (size_t)h * st;
            const float  bh  = beta[h];
            const float  decay = expf(g[h]);
            float *Sh = S + (size_t)h * st * st;
            float *oh = s->o + (size_t)h * st;

            for (int i = 0; i < st; ++i) {
                float *Si = Sh + (size_t)i * st;   /* contiguous row of the state */
                for (int a = 0; a < st; ++a)
                    Si[a] *= decay;
            }

            for (int i = 0; i < st; ++i) {
                float *Si = Sh + (size_t)i * st;
                float sk = 0.0f;                    /* sum over the key axis a */
                for (int a = 0; a < st; ++a)
                    sk += Si[a] * khv[a];

                const float d = bh * (vhv[i] - sk); /* free index = value axis */

                float o = 0.0f;
                for (int a = 0; a < st; ++a) {
                    Si[a] += khv[a] * d;
                    o += Si[a] * qhv[a];
                }
                oh[i] = o * qscale;
            }
        }

        /* ---- gated per-head RMSNorm (48 independent heads) ---- */
        #pragma omp for schedule(static)
        for (int h = 0; h < vh; ++h) {
            float *oh = s->o + (size_t)h * st;
            rmsnorm(oh, oh, ssm_norm, st, m->rms_eps);
        }

        /* silu(z)*o with the ssm_out head permutation (gdn_v_grouped:
         * h' = rep*(h % kh) + h / kh, i.e. entry hp takes head
         * h = kh*(hp % rep) + hp / rep). Iterations are independent. */
        #pragma omp for schedule(static)
        for (int hp = 0; hp < vh; ++hp) {
            const int h = kh * (hp % rep) + hp / rep;
            for (int d = 0; d < st; ++d)
                s->attn_out[(size_t)hp * st + d] =
                    f_silu(s->z[(size_t)h * st + d]) * s->o[(size_t)h * st + d];
        }
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
    (void)pos;   /* RoPE tables now come from bonsai_forward, computed once
                  * per token; pos is no longer needed here */
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

    const bonsai_wcache *wc = &m->wc;
    const size_t aslot = (size_t)attn_slot(m, il);
    rmsnorm(s->xb, s->x, wc->attn_norm + (size_t)il * n_embd, n_embd,
            m->rms_eps);

    /* attn_q/k/v share the SAME input (s->xb) and sign vector: stage the
     * sign+Hadamard transform once and run all three matvecs against it
     * (two of the three Hadamards of this group were pure recompute). */
    if (mm_stage(m, s->xb, m->signs_5120, n_embd) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", il);
    if (mm_run(m, s->qg, nm, n_embd, 2 * qdim) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", il);
    if (mm_run(m, s->k_full, nm, n_embd, nkv * hd) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", il);
    if (mm_run(m, s->v_full, nm, n_embd, nkv * hd) != 0) return -1;

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
    const float *attn_qn = wc->attn_qn + aslot * hd;
    const float *attn_kn = wc->attn_kn + aslot * hd;
    /* RoPE tables are computed once per TOKEN in bonsai_forward (they depend
     * only on pos) -- block_attn just consumes them. */
    const float *rope_cos = s->rope_cos, *rope_sin = s->rope_sin;
    const int nhalf = m->n_rot / 2;
    const int t = m->st.kv_len;
    /* Compacted KV: this block owns row attn_slot(m, il) == aslot, not il
     * (see model.h), because the GDN blocks never touch the cache. */
    float *kc = m->st.kv_cache + (aslot * m->ctx + (size_t)t) * 2 * nkv * hd;
    float *vc = kc + (size_t)nkv * hd;
    const int nt = t + 1;
    const float inv_sqrt_hd = 1.0f / sqrtf((float)hd);

    /* ONE parallel region per attention layer. The old code ran five
     * serial stages here (qg split, 28 QK-norms, rope, kv append) plus a
     * serial 24-head attention loop -- the engine's ONLY O(ctx) serial path
     * (static analysis: ~35-40 us x nt per token, dominating glue as ctx
     * grows) -- and a serial 6144-wide sigmoid gate. Per-head arithmetic
     * order is unchanged and heads write disjoint slices (scores now live in
     * the previously allocated-but-unused s->att[n_head][ctx]), so this is
     * bit-identical; without OpenMP the body runs top-to-bottom in the old
     * order. */
    #pragma omp parallel
    {
        /* split the per-head Q|gate layout of s->qg (see the layout comment
         * above -- head h is {Q(hd), gate(hd)} consecutive in s->qg) */
        #pragma omp for schedule(static)
        for (int h = 0; h < nh; ++h) {
            memcpy(s->q_full + (size_t)h * hd, s->qg + (size_t)h * 2 * hd,
                   (size_t)hd * sizeof(float));
            memcpy(s->attn_gate + (size_t)h * hd,
                   s->qg + (size_t)h * 2 * hd + hd, (size_t)hd * sizeof(float));
        }

        /* per-head QK RMSNorm */
        #pragma omp for schedule(static)
        for (int h = 0; h < nh + nkv; ++h) {
            if (h < nh)
                rmsnorm(s->q_full + (size_t)h * hd, s->q_full + (size_t)h * hd,
                        attn_qn, hd, m->rms_eps);
            else
                rmsnorm(s->k_full + (size_t)(h - nh) * hd,
                        s->k_full + (size_t)(h - nh) * hd, attn_kn, hd,
                        m->rms_eps);
        }

        /* M-RoPE per head (NeoX pairing, first n_rot dims) */
        #pragma omp for schedule(static)
        for (int h = 0; h < nh + nkv; ++h) {
            float *v = (h < nh) ? s->q_full + (size_t)h * hd
                                : s->k_full + (size_t)(h - nh) * hd;
            rope_apply_head(v, nhalf, rope_cos, rope_sin);
        }

        /* append to the KV cache (kv_len advances in bonsai_forward) */
        #pragma omp single
        {
            memcpy(kc, s->k_full, (size_t)nkv * hd * sizeof(float));
            memcpy(vc, s->v_full, (size_t)nkv * hd * sizeof(float));
        }

        /* attention heads, each with its own scores row */
        #pragma omp for schedule(static)
        for (int h = 0; h < nh; ++h) {
            const int hkv = h / rep;
            const float *qh = s->q_full + (size_t)h * hd;
            float *sc = s->att + (size_t)h * m->ctx;
            float mx = -INFINITY;
            for (int p = 0; p < nt; ++p) {
                const float *kp = m->st.kv_cache +
                    (aslot * m->ctx + (size_t)p) * 2 * nkv * hd + (size_t)hkv * hd;
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
                    (aslot * m->ctx + (size_t)p) * 2 * nkv * hd + (size_t)nkv * hd +
                    (size_t)hkv * hd;
                for (int d = 0; d < hd; ++d) oh[d] += w * vp[d];
            }
        }

        /* gate: sigmoid over the second half of the fused Q projection */
        #pragma omp for schedule(static)
        for (int i = 0; i < qdim; ++i)
            s->attn_out[i] = s->q_full[i] * f_sigmoid(s->attn_gate[i]);
    }

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

    rmsnorm(s->xb, s->x, m->wc.post_norm + (size_t)il * m->n_embd, m->n_embd,
            m->rms_eps);

    /* ffn_gate and ffn_up share the SAME input (s->xb) and sign vector: stage
     * the sign+Hadamard transform once and run both matvecs against it. */
    if (mm_stage(m, s->xb, m->signs_5120, m->n_embd) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il);
    if (mm_run(m, s->hb, nm, m->n_embd, m->n_ff) != 0) return -1;
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", il);
    if (mm_run(m, s->hb2, nm, m->n_embd, m->n_ff) != 0) return -1;

    /* n_ff independent lanes: hb[i] reads only hb[i] and hb2[i]. */
    #pragma omp parallel for schedule(static)
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

/* want_logits == 0 skips the final RMSNorm + the vocabulary-wide output head.
 * Prefill only needs the logits of its LAST token, so every earlier prompt
 * token can skip the whole n_vocab projection (~5% of prefill).  The KV cache
 * and the recurrent state are unaffected: the head is pure read-out. */
static int forward_body(bonsai_model *m, int32_t token, int32_t pos,
                        int want_logits) {
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

    /* RoPE cos/sin depend only on pos: compute ONCE per token here (the old
     * code recomputed them per attention layer -- 16x the pow/cos/sin). */
    rope_tables_cached(m, pos, s->rope_cos, s->rope_sin);

    /* Every block can fail (missing/mismatched/unsupported weight, bad
     * geometry). Previously those failures were swallowed: mm_folded() returned
     * early leaving `y` stale, so the logits kept the previous token's values
     * while bonsai_forward() still reported success. Now the first failure
     * aborts the token with a non-zero return. */
    for (int il = 0; il < m->n_layer; ++il) {
        const int is_full_attn = ((il + 1) % m->full_attn_interval == 0);
        const int rc = is_full_attn ? block_attn(m, il, pos) : block_gdn(m, il);
        if (rc != 0) {
            fprintf(stderr, "forward: block %d failed\n", il);
            return -1;
        }
        if (block_ffn(m, il) != 0) {
            fprintf(stderr, "forward: block %d ffn failed\n", il);
            return -1;
        }
    }
    m->st.kv_len = pos + 1;

    if (!want_logits)
        return 0;

    /* final norm + output head (folded with the width-5120 signs) */
    rmsnorm(s->xb, s->x, m->wc.output_norm, m->n_embd, m->rms_eps);
    if (mm_folded(m, s->logits, s->xb, "output.weight", m->signs_5120,
                  m->n_embd, m->n_vocab) != 0)
        return -1;
    return 0;
}

int bonsai_forward(bonsai_model *m, int32_t token, int32_t pos) {
    return forward_body(m, token, pos, 1);
}

int bonsai_forward_nologits(bonsai_model *m, int32_t token, int32_t pos) {
    return forward_body(m, token, pos, 0);
}

int bonsai_generate(bonsai_model *m, const int32_t *prompt, int32_t n_prompt,
                    int32_t *out, int32_t n_out) {
    int32_t pos = 0;
    for (int32_t i = 0; i < n_prompt; ++i) {
        if (bonsai_forward(m, prompt[i], pos++) != 0) return -1;
    }
    int32_t produced = 0;
    for (int32_t i = 0; i < n_out; ++i) {
        const int32_t best = bonsai_argmax(m->sc.logits, m->n_vocab);
        out[produced++] = best;
        if (i + 1 < n_out) {
            if (pos >= m->ctx) break;
            if (bonsai_forward(m, best, pos++) != 0) break;
        }
    }
    return produced;
}

/* Greedy argmax over the logits. Scalar on the host (bit-exact baseline);
 * vectorised on the card. Tie rule preserved exactly: `>` keeps the LOWEST
 * index -- per lane the strictly-greater update keeps the first occurrence
 * in that lane's strided subsequence, and the cross-lane reduction picks the
 * lowest index among equal maxima. NaNs never win in either path. */
int32_t bonsai_argmax(const float *x, int32_t n) {
#ifdef __MIC__
    if (n >= 16 && (n & 15) == 0 && ((uintptr_t)x & 63u) == 0) {
        const __m512i lane =
            _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        __m512 vmax = _mm512_set1_ps(-INFINITY);
        __m512i vidx = _mm512_setzero_si512();
        for (int32_t i = 0; i < n; i += 16) {
            const __m512 v = _mm512_load_ps(x + i);
            /* IMCI predicate set is _OS/_US-flavoured (no _CMP_GT_OQ/_CMP_LT_OQ):
             * vmax < v is the same mask; NaN -> false, matching scalar `>`. */
            const __mmask16 msk = _mm512_cmp_ps_mask(vmax, v, _CMP_LT_OS);
            vmax = _mm512_mask_mov_ps(vmax, msk, v);
            vidx = _mm512_mask_mov_epi32(vidx, msk,
                       _mm512_add_epi32(_mm512_set1_epi32(i), lane));
        }
        float bv[16] __attribute__((aligned(64)));
        int32_t bi[16] __attribute__((aligned(64)));
        _mm512_store_ps(bv, vmax);
        _mm512_store_si512(bi, vidx);
        float bestv = bv[0];
        int32_t besti = bi[0];
        for (int j = 1; j < 16; ++j)
            if (bv[j] > bestv || (bv[j] == bestv && bi[j] < besti)) {
                bestv = bv[j]; besti = bi[j];
            }
        return besti;
    }
#endif
    {
        int32_t best = 0;
        float bv = x[0];
        for (int32_t v = 1; v < n; ++v)
            if (x[v] > bv) { bv = x[v]; best = v; }
        return best;
    }
}
