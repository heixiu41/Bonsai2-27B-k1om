/*
 * model.h - Bonsai 2 27B (qwen35) forward pass for the Xeon Phi 7120P.
 *
 * Architecture (all values read from the GGUF at runtime, defaults shown):
 *
 *   64 blocks, embedding 5120, SwiGLU FFN 17408, vocab 248320
 *   Every 4th block (i%4==3) is FULL ATTENTION, the other 48 are the Gated
 *   Delta Net (linear attention):
 *     full : 24 heads / 4 kv heads / head_dim 256, QK-RMSNorm, M-RoPE n_rot=64
 *     GDN  : 16 key heads (2048) + 48 value heads (6144), state 128x128,
 *            causal depthwise conv kernel 4, L2-normalised q/k, sigmoid beta,
 *            softplus alpha, per-head gated RMSNorm
 *
 * Memory strategy (16 GB card, 7.2 GB model):
 *   Weights stay in their on-disk PQ2_0 form (2.125 bpw) and are decoded
 *   128 weights at a time inside the matmul, so the resident cost of the big
 *   folded weights is the file size itself. See pq2_0.h. The exception is the
 *   small per-layer constant tables (norms, ssm_beta/alpha, conv1d, ...):
 *   they are token-invariant and are dequantised ONCE at load time into
 *   bonsai_wcache below (~110 MB), instead of being re-dequantised every
 *   token.
 *
 * Numerical strategy:
 *   The reference implementation (scripts/bonsai_forward.py) is the oracle.
 *   Layer constants here are written to mirror it exactly.
 */
#ifndef BONSAI_MODEL_H
#define BONSAI_MODEL_H

#include "gguf.h"
#include "pq2_0.h"

#include <stdint.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* per-layer recurrent state                                           */
/* ------------------------------------------------------------------ */
typedef struct {
    /* GDN: state matrix per value head, 128x128, decayed every token */
    float *ssm_state;          /* [n_layer][n_v_heads][state][state] */

    /* GDN: causal conv history, (kernel-1) previous QKV vectors */
    float *conv_hist;          /* [n_layer][conv_k-1][conv_dim] */

    /* full attention ONLY: KV cache, [n_attn][ctx][2][n_head_kv][head_dim].
     * Compacted on purpose -- the n_layer-n_attn GDN blocks are linear
     * attention and never touch it, so the old n_layer stride wasted 4x here
     * (3 GiB at ctx=8192).  Index it with model.c:attn_slot(m, il). */
    float *kv_cache;
    int32_t kv_len;            /* tokens currently cached */
} bonsai_state;

/* ------------------------------------------------------------------ */
/* scratch buffers. Sized once from the config; nothing is allocated in  */
/* the per-token loop except through these.                             */
/* ------------------------------------------------------------------ */
typedef struct {
    float *x, *xb, *xb2;       /* [n_embd]                        */
    float *qkv;                /* [conv_dim] = 2*key_dim+value_dim */
    float *z;                  /* [value_dim]                      */
    float *conv_out;           /* [conv_dim]                       */
    float *q, *k;              /* [n_v_heads][state]  (after repeat) */
    float *v;                  /* [n_v_heads][state]               */
    float *o;                  /* [n_v_heads][state] DeltaNet out  */
    float *delta;              /* [n_v_heads][state]               */
    float *beta_out;           /* [2*n_v_heads] beta then alpha    */
    float *attn_out;           /* [value_dim]                      */
    float *hb, *hb2;           /* [n_ff]                           */
    float *logits;             /* [n_vocab]                        */

    float *qg;                 /* [2*n_head*head_dim]  per head: Q|gate */
    float *q_full, *k_full, *v_full;
    float *attn_gate;          /* [n_head*head_dim]  gate half, per-head layout */
    float *att;                /* [n_head][ctx] scores             */
    float *rope_cos, *rope_sin;/* [n_rot/2]                        */
    float *scores;             /* [ctx]                            */

    /* scratch for the Hadamard transform of a matmul input. Must hold the
     * widest folded input (17408 for ffn_down). */
    float *fold_x;             /* [17408]                          */
    /* permuted (xp) form of fold_x, built once per mm_stage on k1om: the 2-3
     * matvecs sharing a staged input reuse it instead of each rebuilding
     * the permutation (401 -> ~258 builds per token). */
    float *fold_xp;            /* [17408]                          */
} bonsai_scratch;

/* ------------------------------------------------------------------ */
/* token-invariant weight cache                                        */
/*                                                                     */
/* The small per-layer constant weights below used to be re-dequantised*/
/* from the GGUF on EVERY token (the ssm_beta/ssm_alpha pair alone is  */
/* ~23.6M BF16->F32 conversions per token, measured ~85 ms/token on the*/
/* card). They are now dequantised ONCE in bonsai_model_open() through */
/* the very same dequant_to_f32() path, so every cached float is       */
/* bit-identical to the value the old per-token code produced, and the */
/* block functions only do pointer lookups. GDN-only arrays are indexed*/
/* by the GDN slot g = il - (il+1)/interval and attention-only ones by */
/* a = (il+1)/interval - 1, so no memory is spent on the wrong block   */
/* type. ~110 MB total, fine on the 15.5 GB card.                      */
/* ------------------------------------------------------------------ */
typedef struct {
    float *attn_norm;   /* [n_layer][n_embd]                          */
    float *post_norm;   /* [n_layer][n_embd]  post_attention_norm     */
    float *output_norm; /* [n_embd]                                   */

    /* GDN blocks only (48 of 64) */
    float *ssm_beta;    /* [n_gdn][n_embd * n_v_heads]  BF16 source   */
    float *ssm_alpha;   /* [n_gdn][n_embd * n_v_heads]  BF16 source   */
    float *ssm_a;       /* [n_gdn][n_v_heads]           F32           */
    float *ssm_dt;      /* [n_gdn][n_v_heads]           F32 bias      */
    float *ssm_norm;    /* [n_gdn][head_v_dim]          F32           */
    float *ssm_conv1d;  /* [n_gdn][conv_dim * conv_k]   F32           */

    /* full-attention blocks only (16 of 64) */
    float *attn_qn;     /* [n_attn][head_dim]           F32           */
    float *attn_kn;     /* [n_attn][head_dim]           F32           */
} bonsai_wcache;

/* ------------------------------------------------------------------ */
/* model handle                                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    bonsai_gguf   *g;
    bonsai_state   st;
    bonsai_scratch sc;
    bonsai_wcache  wc;

    /* geometry (mirrors the reference implementation's fields) */
    int32_t n_layer, n_embd, n_ff, n_vocab;
    int32_t n_head, n_head_kv, head_dim;
    int32_t n_rot;
    int32_t ctx;                 /* context length actually allocated */

    int32_t ssm_d_conv, ssm_d_state, ssm_inner, ssm_dt_rank, ssm_group;
    int32_t key_dim, value_dim, conv_dim, head_v_dim;

    int32_t full_attn_interval;
    int32_t n_attn;              /* full-attention blocks = layers with
                                  * (il+1) % full_attn_interval == 0; the other
                                  * n_layer-n_attn are GDN (linear) blocks */
    int32_t had_block;
    float   rms_eps;
    double  rope_freq_base;
    double *rope_freq;          /* [n_rot/2] freq_base^(-2i/n_rot), cached at open */

    /* Hadamard activation transform: sign vectors keyed by input width */
    float *signs_5120, *signs_6144, *signs_17408;

    /* decoded token ids waiting to be processed (prompt prefill) */
    int32_t  *prompt;
    int32_t   prompt_len;

    /* Optional debug tap. When `debug` is non-NULL, bonsai_forward() copies
     * the named intermediate into `debug` at the matching point:
     *   "embed"     after the embedding row + inverse transform
     *   "attn_norm" after RMSNorm(attn_norm) of the current block
     *   "qkv"       after W_qkv @ had(attn_norm)      (GDN blocks)
     *   "conv"      after silu(conv)                  (GDN blocks)
     *   "block_out" after the block's residual add
     * `debug_layer` selects which block. */
    float   *debug;
    int32_t  debug_cap;
    int32_t  debug_len;
    const char *debug_want;
    int32_t  debug_layer;
} bonsai_model;

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* Open the model, read its metadata/sign tables and allocate state.
 * `ctx` is the maximum sequence length; pass 0 for a memory-frugal default. */
bonsai_model *bonsai_model_open(const char *gguf_path, int32_t ctx);

void bonsai_model_close(bonsai_model *m);

/* Bytes of the recurrent state + scratch + sign tables (excludes the file). */
uint64_t bonsai_model_runtime_bytes(const bonsai_model *m);

/* Print geometry and both the weight and runtime memory budgets. */
void bonsai_model_describe(const bonsai_model *m);

/* ------------------------------------------------------------------ */
/* inference                                                           */
/* ------------------------------------------------------------------ */

/* Reset all recurrent state and the KV cache (does not reload weights). */
void bonsai_reset(bonsai_model *m);

/* Run one token at position `pos`, filling m->sc.logits.
 * Returns 0 on success. */
int bonsai_forward(bonsai_model *m, int32_t token, int32_t pos);
/* Same state update and attention work, but skips the final RMSNorm + the
 * n_vocab-wide output head.  Prefill only needs the logits of its LAST token,
 * so the earlier prompt tokens can use this (~5% of prefill); m->sc.logits is
 * left untouched, so do NOT read it after a nologits call. */
int bonsai_forward_nologits(bonsai_model *m, int32_t token, int32_t pos);

/* Greedy decode from a token sequence. Writes `n_out` ids into `out`.
 * Returns the number actually produced. */
int bonsai_generate(bonsai_model *m, const int32_t *prompt, int32_t n_prompt,
                    int32_t *out, int32_t n_out);

/* Greedy argmax over n floats (ties keep the lowest index). Vectorised on
 * k1om, scalar elsewhere. */
int32_t bonsai_argmax(const float *x, int32_t n);

#endif /* BONSAI_MODEL_H */
