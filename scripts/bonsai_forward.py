"""Ternary Bonsai 2 27B - reference forward pass (numpy, correctness-first).

Implements the qwen35 hybrid architecture as specified in
docs/BONSAI2_FORWARD_SPEC.md, which was extracted line-by-line from the
PrismML fork. Deliberately slow and readable: this is the correctness oracle
that the k1om port is validated against.

Notes:
  * Tensors are fetched from the remote GGUF over HTTP Range requests and
    dequantised to fp32 on demand, so peak RSS stays bounded instead of
    expanding the whole 7.2 GB model at once.
  * The Hadamard transform is applied to activations for the 401 folded
    weights, and inverted after the embedding lookup, exactly as the fork's
    build_lora_mm does it.

Usage:
    python scripts/bonsai_forward.py --selfcheck
    python scripts/bonsai_forward.py --tokens-csv 1,2,3 --tokens 4
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict

import numpy as np

from bonsai_ref import (Gguf, apply_hadamard, wht_matrix, PQ2_FILE,
                        GGML_F32, GGML_F16, GGML_BF16, GGML_PQ2_0, GGML_PTQ1_0,
                        _dequant_pq2_0, _dequant_ptq1_0)


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------
def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm over the last axis, as ggml's LLM_NORM_RMS does."""
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    # stable form: log1p(exp(-|x|)) + max(x, 0)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / np.maximum(n, eps)


def rope_multi(x: np.ndarray, pos: int, freq_base: float, n_rot: int):
    """M-RoPE over the first n_rot dims of the last axis.

    Pairing is NeoX / "rotate_half": dim i is paired with dim i + n_rot/2, i.e.
    (i, i+32) for n_rot=64 -- NOT the adjacent pair (2i, 2i+1).  This mirrors
    ggml's MROPE path, rotate_pairs(n_dims, n_offset = n_dims/2, scale = 2)
    (ggml-cpu/ops.cpp:6091-6093), and `rope_apply()` in src/bonsai/model.c.
    Adjacent pairing is a different transform: it changes the hidden state
    (first3 error at l_out-63 grows from 4.6e-2 to 1.0e-1) while leaving the
    top-1 token unchanged, so it cannot be caught by a top-1 check.

    For text-only input every section carries the same position, so the
    section split (dimension_sections=[11,11,10,0]) degenerates to standard
    RoPE with theta_i = pos * freq_base^(-2i/n_rot), i in [0, n_rot/2).

    Only the first n_rot dims are touched; dims [n_rot, head_dim) pass through
    bit-identical (head_dim is 256 here, n_rot is 64).
    """
    out = x.copy()
    nr = n_rot // 2
    i = np.arange(nr, dtype=np.float64)
    ang = pos * (freq_base ** (-2.0 * i / n_rot))
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    v0, v1 = x[..., :nr].copy(), x[..., nr:n_rot].copy()
    out[..., :nr] = v0 * cos - v1 * sin
    out[..., nr:n_rot] = v0 * sin + v1 * cos
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class BonsaiForward:
    def __init__(self, max_cache: int = 160, verbose: bool = False,
                 local: str | None = None):
        t0 = time.time()
        self.g = Gguf.open(PQ2_FILE, local=local)
        m = self.g.meta
        self.n_embd = int(m["qwen35.embedding_length"])
        self.n_layer = int(m["qwen35.block_count"])
        self.n_ff = int(m["qwen35.feed_forward_length"])
        self.n_vocab = self.g.tensors["token_embd.weight"].dims[1]
        self.eps = float(m["qwen35.attention.layer_norm_rms_epsilon"])
        self.rope_base = float(m["qwen35.rope.freq_base"])
        self.n_rot = int(m["qwen35.rope.dimension_count"])
        self.sections = list(m["qwen35.rope.dimension_sections"])

        self.n_head = int(m["qwen35.attention.head_count"])
        self.n_head_kv = int(m["qwen35.attention.head_count_kv"])
        self.head_dim = int(m["qwen35.attention.key_length"])

        self.state = int(m["qwen35.ssm.state_size"])
        self.n_k_heads = int(m["qwen35.ssm.group_count"])
        self.n_v_heads = int(m["qwen35.ssm.time_step_rank"])
        self.key_dim = self.state * self.n_k_heads
        self.value_dim = self.state * self.n_v_heads
        self.conv_k = int(m["qwen35.ssm.conv_kernel"])
        self.full_attn_interval = int(m["qwen35.full_attention_interval"])
        self.is_recr = [((i + 1) % self.full_attn_interval != 0)
                        for i in range(self.n_layer)]

        plan = self.g.hadamard_plan()
        self.had_block = plan["block_size"]
        self.had_signs = plan["signs"]
        self.had_fwd = plan["forward"]
        self.had_inv = set(plan["inverse"])
        self.H = wht_matrix(self.had_block)

        self.ssm_state = [np.zeros((self.n_v_heads, self.state, self.state),
                                   dtype=np.float32)
                          for _ in range(self.n_layer)]
        self.conv_hist = [np.zeros((self.conv_k - 1,
                                    self.key_dim * 2 + self.value_dim),
                                   dtype=np.float32)
                          for _ in range(self.n_layer)]
        self.kv = [None] * self.n_layer

        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._max_cache = max_cache
        self._erows = None
        self.verbose = verbose
        if verbose:
            print(f"[init] {time.time() - t0:.1f}s vocab={self.n_vocab} "
                  f"layers={self.n_layer} recr={sum(self.is_recr)}")

    # -- HTTP with retry -------------------------------------------------
    def _get(self, url: str, headers: dict, timeout: int):
        last = None
        for attempt in range(3):
            try:
                r = self.g._http.get(url, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r
            except Exception as exc:                      # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise last

    # -- weights ---------------------------------------------------------
    def W(self, name: str) -> np.ndarray:
        """Dequantised weight in numpy convention (out, in)."""
        got = self._cache.get(name)
        if got is not None:
            self._cache.move_to_end(name)
            return got
        t = self.g.tensors[name]
        start = self.g.data_base + t.offset
        raw = self._get(self.g.url,
                        {"Range": f"bytes={start}-{start + t.nbytes - 1}"},
                        900).content
        if len(raw) != t.nbytes:
            raise IOError(f"{name}: got {len(raw)} want {t.nbytes}")
        shape = tuple(reversed(t.dims))
        if t.type == GGML_F32:
            a = np.frombuffer(raw, dtype="<f4").reshape(shape)
        elif t.type == GGML_F16:
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        elif t.type == GGML_BF16:
            a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                 ).view(np.float32).reshape(shape)
        elif t.type == GGML_PQ2_0:
            a = _dequant_pq2_0(raw, shape)
        elif t.type == GGML_PTQ1_0:
            a = _dequant_ptq1_0(raw, shape)
        else:
            raise NotImplementedError(t.type_name)
        self._cache[name] = a
        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return a

    def had(self, name: str, x: np.ndarray) -> np.ndarray:
        w = self.had_fwd.get(name)
        if w is None:
            return x
        return apply_hadamard(x, self.had_signs[w], self.had_block)

    def lin(self, name: str, x: np.ndarray) -> np.ndarray:
        """y = W @ had(x)."""
        return self.W(name) @ self.had(name, x)

    # -- layers ----------------------------------------------------------
    def layer_ssm(self, il: int, x: np.ndarray, pos: int) -> np.ndarray:
        p = f"blk.{il}."
        xb = rmsnorm(x, self.W(p + "attn_norm.weight"), self.eps)

        qkv = self.lin(p + "attn_qkv.weight", xb)      # [10240]
        z = self.lin(p + "attn_gate.weight", xb)       # [6144]

        beta = sigmoid(self.W(p + "ssm_beta.weight") @ xb)           # [48]
        alpha = softplus(self.W(p + "ssm_alpha.weight") @ xb
                         + self.W(p + "ssm_dt.bias"))                # [48]
        g = alpha * self.W(p + "ssm_a")                              # [48]

        # causal depthwise conv over the QKV channels, carrying 3 history steps
        win = np.concatenate([self.conv_hist[il], qkv[None, :]], axis=0)
        self.conv_hist[il] = win[1:]
        kern = self.W(p + "ssm_conv1d.weight")          # (C, K)
        conv = silu(np.sum(kern * win.T, axis=1))       # [C]

        q = conv[:self.key_dim].reshape(self.n_k_heads, self.state)
        k = conv[self.key_dim:2 * self.key_dim].reshape(self.n_k_heads, self.state)
        v = conv[2 * self.key_dim:].reshape(self.n_v_heads, self.state)

        qk = l2norm(np.concatenate([q, k], axis=0))
        q, k = qk[:self.n_k_heads], qk[self.n_k_heads:]

        rep = self.n_v_heads // self.n_k_heads          # 3
        # v head h is fed by k/q head h % n_k_heads -- that is what ggml_repeat
        # does and what the fused kernel uses (iq1 = iv1 % neq1).  np.repeat
        # (which gives h // rep) is a different permutation and scrambles most
        # of the 48 value heads.
        qv = np.tile(q, (rep, 1)) * (1.0 / np.sqrt(self.state))
        kv = np.tile(k, (rep, 1))

        # State is S[i][j] with i = KEY index, j = VALUE index, stored exactly
        # as the fused kernel stores it (M[j*S_v + i] == S[i][j]).  The delta
        # rule reads it along its FIRST axis -- ops.cpp:11082-11099:
        #     delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
        #     S[i][j] += k[i]*delta[j]
        #     o[j]     = (1/sqrt(S_v)) * sum_i S[i][j]*q[i]
        # Contracting the SECOND axis instead (S@k, S@q) mixes the key and
        # value axes and is a different recurrence.
        S = self.ssm_state[il] * np.exp(g)[:, None, None]
        sk = np.einsum("hij,hi->hj", S, kv)             # sum_i S[i][j] k[i]
        d = (v - sk) * beta[:, None]
        S = S + kv[:, :, None] * d[:, None, :]          # S[i][j] += k[i]*d[j]
        o = np.einsum("hij,hi->hj", S, qv).reshape(-1)  # o[j] = sum_i S[i][j] q[i]
        self.ssm_state[il] = S

        # gated normalisation is per v-head (head dim = 128)
        oh = rmsnorm(o.reshape(self.n_v_heads, self.state),
                     self.W(p + "ssm_norm.weight"), self.eps).reshape(-1)
        y = silu(z) * oh

        # prism.hadamard.gdn_v_grouped: ONLY the .ssm_out. weights see the 48
        # value heads re-ordered from the tiled [hd, n_k, rep] feature order to
        # the grouped [hd, rep, n_k] one -- before the sign vector and the
        # blockwise Hadamard, not after (llama-model.cpp:2123-2132 sets
        # perm_hd/perm_nk/perm_rep for ".ssm_out." names, llama-graph.cpp:1561-1568
        # reshapes/permutes).  On the head index the net effect is
        #     h -> 3*(h % n_k_heads) + (h // n_k_heads)
        # and without it every GDN layer's output projection is wrong.
        y = y.reshape(rep, self.n_k_heads, self.state) \
             .transpose(1, 0, 2).reshape(self.n_v_heads * self.state)
        return self.lin(p + "ssm_out.weight", y)

    def layer_attn(self, il: int, x: np.ndarray, pos: int) -> np.ndarray:
        p = f"blk.{il}."
        xb = rmsnorm(x, self.W(p + "attn_norm.weight"), self.eps)

        qg = self.lin(p + "attn_q.weight", xb)          # [12288]
        hd = self.head_dim
        # The 12288-wide projection is NOT [Q(6144) | gate(6144)].  The official
        # graph reshapes it {256, 24} for Q and REUSES the same view as the gate,
        # so the layout is per head: [n_head, {Q(256), gate(256)}].  Reading
        # contiguous halves instead picks up each head's second query half as the
        # gate, moving sigmoid(gate) from ~0.03 to ~0.6 and inflating the whole
        # attention output ~3x.
        # Verified against the dump (layer 3, token 0):
        #   reshape(24,2,256)[:,1,:] first3 = [-3.2699 -4.5540 -4.1169]
        #   official gate_reshaped-3 first3 = [-3.2760 -4.5583 -4.1098]
        #   sigmoid sum 128.967 vs official 128.947  (contiguous halves: ~1587)
        q_raw = qg.reshape(self.n_head, 2, hd)[:, 0, :]
        gate = qg.reshape(self.n_head, 2, hd)[:, 1, :].reshape(-1)
        k_raw = self.lin(p + "attn_k.weight", xb).reshape(self.n_head_kv, hd)
        v_raw = self.lin(p + "attn_v.weight", xb).reshape(self.n_head_kv, hd)

        q = rmsnorm(q_raw, self.W(p + "attn_q_norm.weight"), self.eps)
        k = rmsnorm(k_raw, self.W(p + "attn_k_norm.weight"), self.eps)
        q = rope_multi(q, pos, self.rope_base, self.n_rot)
        k = rope_multi(k, pos, self.rope_base, self.n_rot)

        if self.kv[il] is None:
            self.kv[il] = (k[None], v_raw[None])
        else:
            kc, vc = self.kv[il]
            self.kv[il] = (np.concatenate([kc, k[None]], axis=0),
                           np.concatenate([vc, v_raw[None]], axis=0))
        kc, vc = self.kv[il]

        rep = self.n_head // self.n_head_kv
        kk = np.repeat(kc, rep, axis=1)
        vv = np.repeat(vc, rep, axis=1)

        scores = np.einsum("hd,thd->ht", q, kk) / np.sqrt(hd)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        attn = np.einsum("ht,thd->hd", w, vv).reshape(-1)

        return self.lin(p + "attn_output.weight", attn * sigmoid(gate))

    def layer(self, il: int, x: np.ndarray, pos: int) -> np.ndarray:
        h = (self.layer_ssm(il, x, pos) if self.is_recr[il]
             else self.layer_attn(il, x, pos))
        x = x + h
        p = f"blk.{il}."
        xb = rmsnorm(x, self.W(p + "post_attention_norm.weight"), self.eps)
        act = silu(self.lin(p + "ffn_gate.weight", xb)) * self.lin(p + "ffn_up.weight", xb)
        return x + self.lin(p + "ffn_down.weight", act)

    # -- embedding lookup ------------------------------------------------
    def _load_embed_rows(self, first: int, count: int):
        Wt = self.g.tensors["token_embd.weight"]
        n_embd, n_vocab = Wt.dims
        row_bytes = n_embd // 128 * 34
        first = max(0, first)
        count = min(count, n_vocab - first)
        start = self.g.data_base + Wt.offset + first * row_bytes
        raw = self._get(self.g.url,
                        {"Range": f"bytes={start}-{start + count * row_bytes - 1}"},
                        300).content
        assert len(raw) == count * row_bytes, (len(raw), count * row_bytes)
        return _dequant_pq2_0(raw, (count, n_embd)), first, count

    def embed(self, token: int) -> np.ndarray:
        """Row lookup, then invert the rotated basis.

        Verified against the fork's eval-callback dump for this exact model:
            node_0            = GET_ROWS(token_embd.weight, inp_tokens)
            (reshaped)        = RESHAPE(node_0, {1024, n/1024})
            node_2            = MUL_MAT(prism.hadamard.1024, (reshaped))
            model.input_embed = MUL((reshaped), prism.hadamard.signs.5120)

        Two things this pins down that are easy to get wrong:
          * the inverse DOES carry the sign vector (applied after H), and
          * the sign vector used is the one matching n_embd (5120), applied on
            the flattened 5120-wide row, whereas the H transform is blockwise
            1024.
        `apply_hadamard` multiplies by the explicit matrix whose entries are
        already scaled by 1/sqrt(block), so no extra normalisation is applied.
        """
        Wt = self.g.tensors["token_embd.weight"]
        n_embd = Wt.dims[0]
        if self._erows is None or not (self._erow_lo <= token < self._erow_lo + self._erow_n):
            self._erows, self._erow_lo, self._erow_n = \
                self._load_embed_rows((token // 512) * 512, 512)
        row = self._erows[token - self._erow_lo]
        # H is blockwise over 1024, then the width-matched sign vector
        out = apply_hadamard(row, None, self.had_block)
        return out * self.had_signs[n_embd]

    # -- full forward ----------------------------------------------------
    def logits(self, token: int, pos: int, n_layers=None) -> np.ndarray:
        x = self.embed(token)
        for il in range(n_layers or self.n_layer):
            x = self.layer(il, x, pos)
        x = rmsnorm(x, self.W("output_norm.weight"), self.eps)
        return self.lin("output.weight", x)

    def generate(self, tokens, n_new: int, greedy: bool = True):
        out, pos = [], 0
        lg = None
        for i, t in enumerate(tokens):
            lg = self.logits(t, pos)
            pos += 1
            if self.verbose and i % 8 == 0:
                print(f"  [prefill {i}/{len(tokens)}] top1={int(np.argmax(lg))}")
        nxt = int(np.argmax(lg))
        out.append(nxt)
        for _ in range(n_new - 1):
            lg = self.logits(nxt, pos)
            pos += 1
            out.append(int(np.argmax(lg)))
        return out


# ---------------------------------------------------------------------------
def cmd_selfcheck(args):
    f = BonsaiForward(verbose=True, local=args.local)
    print("\n=== geometry ===")
    print(f"  n_embd={f.n_embd} n_layer={f.n_layer} n_ff={f.n_ff} vocab={f.n_vocab}")
    print(f"  ssm : state={f.state} k_heads={f.n_k_heads} v_heads={f.n_v_heads} "
          f"key_dim={f.key_dim} value_dim={f.value_dim} conv_k={f.conv_k}")
    print(f"  attn: heads={f.n_head} kv_heads={f.n_head_kv} dim={f.head_dim}")
    print(f"  rope: n_rot={f.n_rot} base={f.rope_base:g} sections={f.sections}")
    print(f"  recurrent layers: {sum(f.is_recr)}/{f.n_layer} "
          f"(first full-attn at index "
          f"{next(i for i, r in enumerate(f.is_recr) if not r)})")

    tok = args.token
    x = f.embed(tok)
    print(f"\n=== embedding token {tok} ===")
    print(f"  shape={x.shape} norm={np.linalg.norm(x):.4f} "
          f"range=[{x.min():+.4f}, {x.max():+.4f}]")

    print(f"\n=== per-layer residual trajectory ({args.layers} layers) ===")
    y = x.copy()
    t0 = time.time()
    ok = True
    for il in range(args.layers):
        y = f.layer(il, y, il)
        fin = bool(np.isfinite(y).all())
        ok &= fin
        print(f"  layer {il:2d} ({'ssm ' if f.is_recr[il] else 'attn'}): "
              f"norm={np.linalg.norm(y):9.3f} max|.|={np.abs(y).max():8.3f} "
              f"finite={fin}  [{time.time() - t0:5.1f}s]")

    print("\n=== logits after those layers ===")
    lg = f.logits(tok, 0, n_layers=args.layers)
    print(f"  shape={lg.shape} finite={np.isfinite(lg).all()}")
    print(f"  min={lg.min():.3f} max={lg.max():.3f} std={lg.std():.3f}")
    top = np.argsort(lg)[::-1][:8]
    print(f"  top-8 ids   : {top.tolist()}")
    print(f"  top-8 logits: {[round(float(lg[i]), 3) for i in top]}")
    print("\nSELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_run(args):
    f = BonsaiForward(verbose=args.verbose, local=args.local)
    toks = [int(t) for t in args.tokens_csv.split(",")] if args.tokens_csv else [9707]
    print(f"prompt tokens: {toks}")
    t0 = time.time()
    out = f.generate(toks, args.tokens)
    dt = time.time() - t0
    print(f"\ngenerated {len(out)} tokens in {dt:.1f}s "
          f"({len(out) / max(dt, 1e-9):.4f} tok/s)")
    print(f"token ids: {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", type=int, default=9707)
    ap.add_argument("--layers", type=int, default=8,
                    help="how many layers the self-check runs")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--tokens-csv", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--local", default=None,
                    help="path to a local .gguf (skips all HTTP)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    return (cmd_selfcheck(args) if args.selfcheck else cmd_run(args))


if __name__ == "__main__":
    sys.exit(main())
