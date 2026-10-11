"""Diff the numpy reference forward against llama-eval-callback ground truth.

The dump prints only the first/last 3 values of each dimension, so this script
compares exactly those positions -- no more, no less. Comparing anything else
(concatenating head and tail, or taking the wrong row for a multi-token tensor)
produces convincing-looking false mismatches; that mistake was made twice
already, hence this dedicated tool.

Usage:
    python scripts/bonsai_diff_gt.py                      # layer 0 GDN chain
    python scripts/bonsai_diff_gt.py --tensor attn_norm-0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from bonsai_forward import BonsaiForward, rmsnorm, silu, sigmoid, softplus, l2norm
from bonsai_parse_dump import parse_dump

DUMP = r"D:\bonsai\eval_callback_full.txt"
LOCAL = r"D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf"
PROMPT = [760, 6511, 314, 9338, 369]
TOL = 2e-4          # the dump rounds to 4 decimals


def load_gt(dump: str):
    gt = {}
    for rec in parse_dump(dump):
        gt.setdefault(rec["name"], rec)
    return gt


def cmp_rows(name, mine_by_token, gt_rows, verbose=True):
    """mine_by_token: list of 1-D arrays, one per token. gt_rows: [(head, tail)]."""
    n = min(len(mine_by_token), len(gt_rows))
    worst = 0.0
    for i in range(n):
        g_head, g_tail = gt_rows[i]
        m = mine_by_token[i]
        m_head, m_tail = m[:3], m[-3:]
        e = 0.0
        if g_head and len(m_head) == len(g_head):
            e = max(e, max(abs(a - b) for a, b in zip(g_head, m_head)))
        if g_tail and len(m_tail) == len(g_tail):
            e = max(e, max(abs(a - b) for a, b in zip(g_tail, m_tail)))
        worst = max(worst, e)
    status = "OK" if worst < TOL else "MISMATCH"
    if verbose:
        print(f"  {name:<28} rows={n:<3} worst_err={worst:.3e}  {status}")
    return worst, status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--local", default=LOCAL)
    ap.add_argument("--tensor", default=None)
    args = ap.parse_args(argv)

    f = BonsaiForward(local=args.local)
    gt = load_gt(args.dump)
    print(f"loaded {len(gt)} ground-truth tensors from dump\n")

    if args.tensor:
        rec = gt.get(args.tensor)
        if not rec:
            print(f"{args.tensor} not in dump")
            return 1
        print(f"{args.tensor}: op={rec['op']} dims={rec['dims']} rows={len(rec['rows'])}")
        for i, (h, t) in enumerate(rec["rows"][:6]):
            print(f"  row {i}: head={h} tail={t}")
        return 0

    fails = 0
    print("=== layer 0 GDN chain, per token ===")

    # 1. attn_norm
    mine = [rmsnorm(f.embed(t), f.W("blk.0.attn_norm.weight"), f.eps) for t in PROMPT]
    _, s = cmp_rows("attn_norm-0", mine, gt["attn_norm-0"]["rows"])
    fails += s != "OK"

    # 2. qkv_mixed = W_qkv @ had(attn_norm)
    qkv = [f.lin("blk.0.attn_qkv.weight", m) for m in mine]
    _, s = cmp_rows("linear_attn_qkv_mixed-0", qkv, gt["linear_attn_qkv_mixed-0"]["rows"])
    fails += s != "OK"

    # 3. causal depthwise conv + silu, carrying 3 history steps
    kern = f.W("blk.0.ssm_conv1d.weight")          # (10240, 4)
    hist = np.zeros((3, kern.shape[0]), dtype=np.float32)
    convs = []
    for q in qkv:
        win = np.concatenate([hist, q[None, :]], axis=0)
        hist = win[1:]
        convs.append(silu(np.sum(kern * win.T, axis=1)))
    _, s = cmp_rows("conv_output_silu-0", convs, gt["conv_output_silu-0"]["rows"])
    fails += s != "OK"

    # 4. q/k share one L2 normalisation over the concatenated head groups
    key_dim = f.key_dim
    l2s = []
    for c in convs:
        q = c[:key_dim].reshape(f.n_k_heads, f.state)
        k = c[key_dim:2 * key_dim].reshape(f.n_k_heads, f.state)
        qk = l2norm(np.concatenate([q, k], axis=0))
        l2s.append(qk.reshape(-1))
    _, s = cmp_rows("qk_conv_l2-0 (flat 32x128)", l2s,
                    gt["qk_conv_l2-0"]["rows"])
    fails += s != "OK"

    # 5. delta-net recurrence -> new_state
    S = np.zeros((f.n_v_heads, f.state, f.state), dtype=np.float32)
    rep = f.n_v_heads // f.n_k_heads
    states = []
    outs = []
    for ti, (c, t) in enumerate(zip(convs, PROMPT)):
        xb = mine[ti]
        beta = sigmoid(f.W("blk.0.ssm_beta.weight") @ xb)
        alpha = softplus(f.W("blk.0.ssm_alpha.weight") @ xb + f.W("blk.0.ssm_dt.bias"))
        g = alpha * f.W("blk.0.ssm_a")
        q = c[:key_dim].reshape(f.n_k_heads, f.state)
        k = c[key_dim:2 * key_dim].reshape(f.n_k_heads, f.state)
        v = c[2 * key_dim:].reshape(f.n_v_heads, f.state)
        qk = l2norm(np.concatenate([q, k], axis=0))
        q, k = qk[:f.n_k_heads], qk[f.n_k_heads:]
        qv = np.repeat(q, rep, axis=0) * (1.0 / np.sqrt(f.state))
        kv = np.repeat(k, rep, axis=0)
        S = S * np.exp(g)[:, None, None]
        sk = np.einsum("hij,hj->hi", S, kv)
        d = (v - sk) * beta[:, None]
        S = S + kv[:, :, None] * d[:, None, :]
        o = np.einsum("hij,hj->hi", S, qv)
        outs.append(o)
        states.append(S)

    # new_state is the LAST token's state; the dump lays it out as 128x128x48
    last = states[-1]                       # (48, 128, 128)
    flat = last.transpose(2, 1, 0).reshape(-1)   # ne0=128, ne1=128, ne2=48
    _, s = cmp_rows("new_state-0", [flat], gt["new_state-0"]["rows"])
    fails += s != "OK"

    # 6. gated norm + ssm_out
    zs = [f.lin("blk.0.attn_gate.weight", m) for m in mine]
    _, s = cmp_rows("z-0", zs, gt["z-0"]["rows"])
    fails += s != "OK"

    finals = []
    for o, z in zip(outs, zs):
        oh = rmsnorm(o, f.W("blk.0.ssm_norm.weight")[None, :], f.eps).reshape(-1)
        finals.append(silu(z) * oh)
    _, s = cmp_rows("final_output-0", finals, gt["final_output-0"]["rows"])
    fails += s != "OK"

    outs2 = [f.lin("blk.0.ssm_out.weight", fo) for fo in finals]
    _, s = cmp_rows("linear_attn_out-0", outs2, gt["linear_attn_out-0"]["rows"])
    fails += s != "OK"

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' STAGE(S) MISMATCH'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
