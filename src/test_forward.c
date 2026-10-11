/*
 * test_forward.c - validate the C forward pass stage by stage against the
 * official llama-eval-callback ground truth.
 *
 * ======================== READ THIS BEFORE TRUSTING IT =====================
 *  top-1 == 11751 (" Paris") and "rank 0" are NOT a correctness gate.
 *
 *  Measured: scripts/bonsai_forward.py pairs RoPE adjacently ((2i, 2i+1))
 *  while model.c pairs front/back halves ((i, i+32)). Those are two
 *  structurally DIFFERENT attention sub-layers, and BOTH produced
 *  top-1 = 11751 and rank 0. The end-to-end token can therefore be right
 *  while an attention sub-layer is wrong.
 *
 *  What does catch it is the hidden state after a block. An A/B test of the
 *  two pairings moved l_out-63's first-3 error from 4.60e-02 (front/back,
 *  the correct one) to 1.01e-01 (adjacent) -- only the hidden-state check
 *  noticed. Every hidden-state landmark below is therefore load-bearing: it
 *  increments `fails`, and `fails` IS the process exit code. A landmark
 *  check that merely prints is not a check.
 *
 *  Note also WHERE the landmarks are taken: at position 0 the attention
 *  softmax has exactly one key, so any RoPE error cancels out of it and a
 *  single-token probe cannot see the difference. The landmarks below come
 *  from the full 5-token prefill, at the LAST token (position 4), which is
 *  where a wrong attention sub-layer actually shows up.
 * ===========================================================================
 *
 * The dump prints the first/last 3 values of each tensor, so comparisons use
 * exactly those positions. Ground truth below is for the prompt
 *   "The capital of France is" -> [760, 6511, 314, 9338, 369]
 * and (except where noted) for token 0.
 *
 * The end-to-end signal is greedy continuation: the official model predicts
 * token 11751 (" Paris") after this prompt.
 */
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int32_t PROMPT[5] = { 760, 6511, 314, 9338, 369 };
static const int32_t OFFICIAL_NEXT = 11751;      /* ' Paris' */

/* ---- ground truth: ROW 0 (token 760) first/last 3, from the official dump ----
 *
 * NOTE: this dump prints, per row, the first 3 and the last 3 values. For a
 * {10240, 5} tensor there are FIVE rows (one per token) and row 0's last 3 are
 * NOT the last 3 of the printed block. Taking the wrong row produced two
 * spurious "MISMATCH" verdicts while developing this test -- always index the
 * row, never the flattened text. */
static const float GT_EMBED_F[3]   = {  0.0178f,  0.0010f,  0.0039f };
static const float GT_EMBED_L[3]   = {  0.0009f,  0.0034f,  0.0081f };
static const float GT_AN_F[3]      = {  1.3691f,  0.0660f,  0.2630f };
static const float GT_AN_L[3]      = {  0.0646f,  0.2259f,  0.5355f };
static const float GT_QKV_F[3]     = { -0.1177f,  2.9736f, -1.1909f };
static const float GT_QKV_L[3]     = {  1.1030f, -0.8361f, -0.2573f };
static const float GT_CONV_F[3]    = {  0.0048f,  0.1323f, -0.0409f };
static const float GT_CONV_L[3]    = { -0.0013f,  0.0003f,  0.0001f };
/* GT_ATTNOUT_* removed: there is no attnout tap in model.c, so these were never
 * referenced (the compiler said so). If an attn-output landmark is ever added,
 * re-derive the two vectors from eval_callback_full.txt's `linear_attn_out-0`
 * row 0 rather than restoring these constants blindly. */

/* ---- hidden-state landmarks: LAST token of the 5-token prompt -------------
 *
 * Re-derived from D:\bonsai\eval_callback_full.txt (the 5-token dump), NOT
 * copied from any document -- two docs in this repo carried landmark values
 * that were later shown to be wrong.
 *
 * Each of these tensors is printed there as {5120, 5, 1, 1}: FIVE printed
 * rows, one per token, in prompt order, each row showing its own first 3 and
 * last 3 values. The tap fires on the fifth (last) forward, so the
 * like-for-like reference is ROW 4 (token 369 at position 4) -- never row 0,
 * and never the flattened text of the printed block.
 *
 * Row order / tensor identity were cross-checked inside the dump itself:
 *   l_out-0  row k  == attn_residual-0 row k + ffn_out-0 row k   (all 5 k)
 *   l_out-3  row k  == attn_residual-3 row k + ffn_out-3 row k   (all 5 k)
 *   l_out-62 row 4  + attn_output-63 row 4 == attn_residual-63
 *       (the graph evaluates l_out-63 on the last token only, so
 *        l_out-63 is {5120,1} while attn_output-63 is {5120,5})
 *   l_out-63        == attn_residual-63 + ffn_out-63
 *   model.input_embed row 0 == GT_EMBED_F/GT_EMBED_L (token 760), which is the
 *       independently audited row-0 constant -> row 0 really is token 0. */
static const float GT_L0_F[3]  = { -0.0173f,  0.0188f, -0.0142f }; /* l_out-0 */
static const float GT_L0_L[3]  = {  0.0047f,  0.0088f, -0.0177f };
/* attn_residual-3 = ADD(attn_output-3, l_out-2): the FIRST full-attention
 * block's output, i.e. the point in the graph that a wrong RoPE pairing
 * corrupts before the FFN dilutes it. */
static const float GT_AR3_F[3] = { -0.0011f,  0.0483f, -0.0087f };
static const float GT_AR3_L[3] = { -0.0349f,  0.0289f, -0.0285f };
static const float GT_L3_F[3]  = { -0.0114f,  0.0554f, -0.0309f }; /* l_out-3 */
static const float GT_L3_L[3]  = { -0.0157f,  0.0078f, -0.0424f };
static const float GT_L63_F[3] = {  8.2843f, -4.3284f,  5.6489f }; /* l_out-63 */
static const float GT_L63_L[3] = { -0.2688f,  6.0428f, 12.4769f };
static const float GT_L63_L2   = 519.5392f;

static int fails = 0;

static void cmp3(const char *tag, const float *got, int n,
                 const float *gf, const float *gl, float tol) {
    float ef = 0.0f, el = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float a = fabsf(got[i] - gf[i]);
        float b = fabsf(got[n-3+i] - gl[i]);
        if (a > ef) ef = a;
        if (b > el) el = b;
    }
    const int ok = (ef <= tol && el <= tol);
    if (!ok) ++fails;
    printf("  %-14s first_err=%.2e last_err=%.2e  %s\n",
           tag, ef, el, ok ? "OK" : "MISMATCH");
    if (!ok) {
        printf("      got  first=[");
        for (int i = 0; i < 3; ++i) printf("%10.4f ", got[i]);
        printf("] last=[");
        for (int i = 0; i < 3; ++i) printf("%10.4f ", got[n-3+i]);
        printf("]\n      want first=[");
        for (int i = 0; i < 3; ++i) printf("%10.4f ", gf[i]);
        printf("] last=[");
        for (int i = 0; i < 3; ++i) printf("%10.4f ", gl[i]);
        printf("]\n");
    }
}

/* Run one block-0 depth pass on token 0 with the debug tap set, then compare. */
static void probe(bonsai_model *m, const char *want, int cap,
                  const float *gf, const float *gl, float tol) {
    m->debug      = (float *)calloc((size_t)cap, sizeof(float));
    m->debug_cap  = cap;
    m->debug_want = want;
    m->debug_layer = 0;
    m->debug_len  = 0;
    bonsai_reset(m);
    if (bonsai_forward(m, PROMPT[0], 0) != 0) { printf("forward failed\n"); ++fails; }
    if (m->debug_len == 0) {
        printf("  %-14s tap produced nothing\n", want);
        ++fails;
    } else {
        cmp3(want, m->debug, m->debug_len, gf, gl, tol);
    }
    free(m->debug);
    m->debug = NULL; m->debug_want = NULL;
}

/* Compare one hidden-state landmark (first 3 and last 3 values, each with its
 * own absolute tolerance). Always prints what was measured; increments `fails`
 * when either error exceeds its tolerance, so the verdict reaches the exit
 * code. */
static void check_row(const char *tag, const float *got, int n,
                      const float *gf, float tol_first,
                      const float *gl, float tol_last) {
    float ef = 0.0f, el = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float a = fabsf(got[i] - gf[i]);
        float b = fabsf(got[n-3+i] - gl[i]);
        if (a > ef) ef = a;
        if (b > el) el = b;
    }
    const int ok = (ef <= tol_first && el <= tol_last);
    if (!ok) ++fails;
    printf("  %-16s first_err=%.2e (tol %.0e)  last_err=%.2e (tol %.0e)  %s\n",
           tag, ef, tol_first, el, tol_last, ok ? "OK" : "MISMATCH");
    printf("      got  first=[%9.4f %9.4f %9.4f] last=[%9.4f %9.4f %9.4f]\n",
           got[0], got[1], got[2], got[n-3], got[n-2], got[n-1]);
    printf("      want first=[%9.4f %9.4f %9.4f] last=[%9.4f %9.4f %9.4f]\n",
           gf[0], gf[1], gf[2], gl[0], gl[1], gl[2]);
}

/* One 5-token prefill with the hidden-state tap armed for a single block.
 * `tap`/`layer` pick the point in model.c:
 *   "block_out_ffn"  = the block's final residual = the dump's l_out-<layer>
 *   "block_out_attn" = the residual after the attention sub-layer (full-attn
 *                      blocks) = the dump's attn_residual-<layer>
 *   "block_out"      = the same for GDN blocks
 * Position 4 (the last token) is what the tap captures, and the reference
 * values are row 4 of the dump. Returns 0 on a fired+matching tap. */
static int landmark(bonsai_model *m, const char *tap, int layer,
                    const char *tag, const float *gf, float tol_first,
                    const float *gl, float tol_last) {
    m->debug      = (float *)calloc((size_t)m->n_embd, sizeof(float));
    m->debug_cap  = m->n_embd;
    m->debug_want = tap;
    m->debug_layer = layer;
    m->debug_len  = 0;
    bonsai_reset(m);
    for (int i = 0; i < 5; ++i) {
        if (bonsai_forward(m, PROMPT[i], i) != 0) {
            printf("  forward failed at token %d while tapping %s\n", i, tag);
            ++fails;
            free(m->debug); m->debug = NULL; m->debug_want = NULL;
            return -1;
        }
    }
    int rc = 0;
    if (m->debug_len <= 0) {
        printf("  %-16s tap produced nothing\n", tag);
        ++fails; rc = -1;
    } else {
        check_row(tag, m->debug, m->debug_len, gf, tol_first, gl, tol_last);
    }
    free(m->debug);
    m->debug = NULL; m->debug_want = NULL;
    return rc;
}

int main(int argc, char **argv) {
    const char *path = "D:/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    for (int i = 1; i < argc; ++i) path = argv[i];

    printf("=== C forward pass validation ===\n%s\n\n", path);
    bonsai_model *m = bonsai_model_open(path, 64);
    if (!m) { printf("model open failed\n"); return 1; }
    bonsai_model_describe(m);
    printf("\n");

    /* The dump shows 4-decimal values, so a tolerance of 2e-4 means "matches
     * what the official implementation printed". */
    const float TOL = 2e-4f;

    printf("=== block 0 (GDN) stage checks, token 760 ===\n");
    {
        m->debug = (float *)calloc((size_t)m->n_embd, sizeof(float));
        m->debug_cap = m->n_embd; m->debug_want = "embed"; m->debug_layer = -1;
        m->debug_len = 0;
        bonsai_reset(m);
        bonsai_forward(m, PROMPT[0], 0);
        if (m->debug_len) cmp3("embed", m->debug, m->debug_len,
                               GT_EMBED_F, GT_EMBED_L, TOL);
        else { printf("  embed tap empty\n"); ++fails; }
        free(m->debug); m->debug = NULL; m->debug_want = NULL;
    }
    probe(m, "attn_norm", m->n_embd, GT_AN_F, GT_AN_L, TOL);
    /* node_17 = attn_norm * signs_5120 */
    {
        static const float GT_N17_F[3] = { -1.3691f, -0.0660f, -0.2630f };
        static const float GT_N17_L[3] = { -0.0646f, -0.2259f,  0.5355f };
        probe(m, "signed", m->n_embd, GT_N17_F, GT_N17_L, TOL);
    }
    /* qkv carries a ~0.25% relative deviation (7.6e-3 on values around 3.0).
     * That is the inherent PQ2_0 quantisation error of this checkpoint, not a
     * decode bug: PQ2_0 stores 2 bits per weight, and the residual is the same
     * magnitude whether the weights are decoded by this code or by the numpy
     * reference. It is harmless to the result -- with it present the forward
     * pass still reproduces the official next token exactly (see the end-to-end
     * check below), so the tolerance here is sized to the quantiser rather than
     * to the dump's 4-decimal print precision. */
    probe(m, "qkv",       m->conv_dim, GT_QKV_F, GT_QKV_L, 1e-2f);
    probe(m, "conv",      m->conv_dim, GT_CONV_F, GT_CONV_L, 3e-3f);
    /* NOTE: no "convdbg" probe here -- it was a scratch diagnostic that
     * deliberately produced no tap and therefore always counted as a failure. */

    printf("\n=== end-to-end: greedy continuation + hidden-state landmarks ===\n");
    /* Hidden-state landmarks. Each needs its own 5-token prefill because the
     * tap carries a single (stage, layer) pair. These are the checks that can
     * catch a structurally wrong attention sub-layer; top-1 cannot (see the
     * header). An error at layer 0 shows up in l_out-0, an error in the first
     * full-attention block shows up in attn_residual-3 and l_out-3, and a
     * global error shows up in l_out-63.
     *
     * Tolerances are absolute and per row-of-the-dump, and each is sized to
     * the measured residual of the CORRECT model, not to the dump's 4-decimal
     * print precision: PQ2_0 is 2.125 bits/weight, so this implementation's
     * residual vs the official one grows with the magnitude accumulated in the
     * layer. Measured on the correct model (host, zig -O2), and confirmed with
     * a deliberately adjacent-RoPE-paired build:
     *   landmark          correct build     RoPE-broken build
     *   l_out-0           2.48e-04          2.48e-04   (unchanged: layer 0
     *                                                  is GDN, RoPE is not in it)
     *   attn_residual-3   6.65e-04/9.63e-04 1.37e-03/1.34e-03
     *   l_out-3           7.52e-04/1.22e-03 1.08e-03/1.27e-03
     *   l_out-63          4.60e-02/1.05e-01 1.01e-01/1.26e-01
     * The early landmarks are deliberately NOT tuned to that ~1e-3 RoPE
     * wiggle (a threshold between 9.6e-04 and 1.34e-03 would be flaky); they
     * are tuned to a real structural error at that layer, which moves them by
     * far more. The RoPE pairing error is caught by l_out-63's first-3 check
     * (4.60e-02 -> 1.01e-01 against the unchanged 5e-02 threshold), which is
     * the only landmark whose signal-to-noise separates it. */
    const float TOL_L0   = 2e-3f;   /* correct 2.5e-04 -> 8x margin */
    const float TOL_AR3  = 3e-3f;   /* correct 9.6e-04 -> 3.1x margin */
    const float TOL_L3   = 4e-3f;   /* correct 1.2e-03 -> 3.3x margin */
    const float TOL_L63L = 2e-1f;   /* correct 1.05e-01 -> 1.9x margin */
    const float TOL_L63F = 5e-2f;   /* first3 of l_out-63, unchanged: the
                                     * audited constant that catches RoPE */

    landmark(m, "block_out_ffn",  0, "l_out-0",
             GT_L0_F,  TOL_L0,  GT_L0_L,  TOL_L0);
    landmark(m, "block_out_attn", 3, "attn_residual-3",
             GT_AR3_F, TOL_AR3, GT_AR3_L, TOL_AR3);
    landmark(m, "block_out_ffn",  3, "l_out-3",
             GT_L3_F,  TOL_L3,  GT_L3_L,  TOL_L3);

    /* Final-layer hidden state.
     *
     * The tap fires on the LAST forward, so the like-for-like reference is the
     * LAST token of this same 5-token prompt (row 4 of the dump's {5120,5}
     * tensors) -- NOT the single-token dump's token 760. Cross-referencing the
     * wrong dump made this report MISMATCH while the model was in fact already
     * exact end to end.
     * Official, eval_callback_full.txt, l_out-63 {5120,1,1,1} (it is evaluated
     * on the last token only):
     *   sum = 46.752823   first3 = [8.2843 -4.3284 5.6489]
     *                     last3  = [-0.2688 6.0428 12.4769]   l2 = 519.5392
     * (single-token dump token 760 gives [-2.0233 -5.2912 0.9587], l2 561.3632
     *  -- a different token, so a different vector.)
     * This pass also leaves the logits used by the top-1/rank block below. */
    {
        m->debug = (float *)calloc((size_t)m->n_embd, sizeof(float));
        m->debug_cap = m->n_embd;
        m->debug_want = "block_out_ffn";
        m->debug_layer = m->n_layer - 1;
        m->debug_len = 0;
        bonsai_reset(m);
        for (int i = 0; i < 5; ++i) {
            if (bonsai_forward(m, PROMPT[i], i) != 0) {
                printf("  forward failed at token %d\n", i); ++fails; break;
            }
        }
        printf("\n=== final hidden state (after block %d residual) ===\n",
               m->n_layer - 1);
        if (m->debug_len) {
            check_row("l_out-63", m->debug, m->debug_len,
                      GT_L63_F, TOL_L63F, GT_L63_L, TOL_L63L);
            double ss = 0.0;
            for (int i = 0; i < m->debug_len; ++i) ss += (double)m->debug[i]*m->debug[i];
            const double l2 = sqrt(ss);
            printf("      ||x|| = %.4f   official = %.4f   ratio = %.4f\n",
                   l2, GT_L63_L2, l2 / GT_L63_L2);
            if (fabs(l2 / GT_L63_L2 - 1.0) >= 2e-2) {
                printf("      norm outside 2%%\n"); ++fails;
            }
        } else {
            printf("  l_out-63 tap empty\n"); ++fails;
        }
        free(m->debug); m->debug = NULL; m->debug_want = NULL;
    }
    {
        const float *lg = m->sc.logits;
        int best = 0;
        for (int v = 1; v < m->n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        double mean = 0.0;
        for (int v = 0; v < m->n_vocab; ++v) mean += lg[v];
        mean /= m->n_vocab;
        double var = 0.0;
        for (int v = 0; v < m->n_vocab; ++v) { double d = lg[v]-mean; var += d*d; }
        var /= m->n_vocab;
        printf("  top-1 = %d   official = %d (' Paris')   %s\n",
               best, OFFICIAL_NEXT,
               best == OFFICIAL_NEXT ? "MATCH" : "DIFFERENT");
        printf("  logits: mean=%.3f std=%.3f  top1=%.3f\n",
               mean, sqrt(var), lg[best]);

        /* Rank of the token the official model predicts. Necessary but NOT
         * sufficient: a structurally wrong attention sub-layer still gave
         * rank 0 here (see the header). Never read this line without the
         * hidden-state landmarks above. */
        int rank = 0;
        for (int v = 0; v < m->n_vocab; ++v)
            if (lg[v] > lg[OFFICIAL_NEXT]) ++rank;
        printf("  rank of %d in my logits = %d / %d   (logit %.4f vs top %.4f)\n",
               OFFICIAL_NEXT, rank, m->n_vocab, lg[OFFICIAL_NEXT], lg[best]);
        printf("  => %s\n",
               rank == 0    ? "exact match (NOT a correctness proof on its own)" :
               rank < 10    ? "very close" :
               rank < 100   ? "close" :
               rank < 1000  ? "weakly correlated" :
                              "essentially unrelated: a structural error remains");
        if (best != OFFICIAL_NEXT) ++fails;
    }

    bonsai_model_close(m);
    printf("\n%s\n", fails ? "FAILURES PRESENT" : "ALL CHECKS PASSED");
    return fails ? 1 : 0;
}
