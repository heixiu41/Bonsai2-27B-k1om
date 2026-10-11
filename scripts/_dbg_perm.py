"""Test the missing `gdn_v_grouped` permutation in block 0 (token 760).

Official chain (llama-graph.cpp:1561-1578, only for weights whose name contains
".ssm_out.", llama-model.cpp:2123-2132):

    x = reshape(final_output, [hd=128, n_k=16, rep=3])
    x = cont(permute(x, 0, 2, 1, 3))          -> [128, 3, 16]
    x = reshape(x, 6144)
    x = x * signs.6144
    x = hadamard.1024 blockwise(x)
    linear_attn_out = ssm_out.weight @ x

i.e. the 48 value heads are re-ordered  h' = 3*(h % 16) + (h // 16)  (and the
inverse h = 16*(h' % 3) + (h' // 3)).  Neither scripts/bonsai_forward.py nor
src/bonsai/model.c applies this.

Everything else is the already-patched (orientation + h%16) recurrence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bonsai_forward import BonsaiForward, rmsnorm, silu, softplus, sigmoid, l2norm  # noqa: E402

LOCAL = r"D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf"
SV = 128
T = {
    "attn_output-0": 2.820900, "norm-0 (rms o)": 515.195801, "node_56 (rms*w)": 428.109375,
    "final_output-0 (pre-perm)": -11.738554, "node_63": -2.755580, "node_65": 0.183359,
    "linear_attn_out-0": 6.655458, "attn_residual-0": 7.265354,
    "post norm-0": 62.510811, "ffn_up-0": -4.755027, "ffn_swiglu-0": 0.708571,
    "node_79": -0.775956, "node_81": -2.622696, "ffn_out-0": 2.998708,
    "l_out-0": 10.264053, "norm-1 (rms l_out)": 51.561306,
}
F3 = {
    "linear_attn_out-0": ([-0.0202, 0.0285, -0.0249], [-0.0199, 0.0048, -0.0283]),
    "attn_residual-0": ([-0.0023, 0.0294, -0.0210], [-0.0189, 0.0081, -0.0201]),
    "l_out-0": ([-0.0004, 0.0492, -0.0335], [-0.0121, -0.0091, -0.0611]),
    "ffn_out-0": ([0.0020, 0.0197, -0.0125], [0.0069, -0.0172, -0.0409]),
    "ffn_swiglu-0": ([-0.0069, 0.0146, -0.0030], [0.0015, -0.0007, -0.0123]),
    "final_output-0 (pre-perm)": ([0.0000, -0.0027, -0.0028], [0.0000, 0.0000, -0.0000]),
}
# official RMS recovered exactly from the dump via norm_{N+1} = RMS_NORM(l_out_N)
OFF_RMS = {"attn_residual-0": 8.31, "l_out-0": 14.27}


def st(name, x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    s = float(a.sum())
    l2 = float(np.sqrt((a * a).sum()))
    t = T.get(name)
    extra = f"  OFF_sum={t:13.6f} ratio={s / t:8.4f}" if t is not None else ""
    if name in OFF_RMS:
        extra += f"  l2={l2:8.4f} OFF_l2={OFF_RMS[name]:8.4f} ratio={l2 / OFF_RMS[name]:6.3f}"
    print(f"{name:26s} n={a.size:6d} sum={s:13.6f} l2={l2:10.5f} max|.|={np.abs(a).max():8.4f}{extra}")
    if name in F3:
        f3, l3 = F3[name]
        print(f"{'':26s} first3={np.round(a[:3], 4)} OFF={f3}  last3={np.round(a[-3:], 4)} OFF={l3}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=LOCAL)
    ap.add_argument("--token", type=int, default=760)
    ap.add_argument("--no-perm", action="store_true")
    args = ap.parse_args()
    f = BonsaiForward(local=args.local)
    p = "blk.0."
    x = f.embed(args.token)
    xb = rmsnorm(x, f.W(p + "attn_norm.weight"), f.eps)
    qkv = f.lin(p + "attn_qkv.weight", xb)
    z = f.lin(p + "attn_gate.weight", xb)
    beta = sigmoid(f.W(p + "ssm_beta.weight") @ xb)
    g = softplus(f.W(p + "ssm_alpha.weight") @ xb + f.W(p + "ssm_dt.bias")) * f.W(p + "ssm_a")
    kern = f.W(p + "ssm_conv1d.weight")
    win = np.concatenate([np.zeros((f.conv_k - 1, qkv.size), np.float32), qkv[None, :]], 0)
    conv = silu(np.sum(kern * win.T, axis=1))
    q, k, v = (conv[:f.key_dim].reshape(f.n_k_heads, SV),
               conv[f.key_dim:2 * f.key_dim].reshape(f.n_k_heads, SV),
               conv[2 * f.key_dim:].reshape(f.n_v_heads, SV))
    qk = l2norm(np.concatenate([q, k], axis=0))
    q, k = qk[:f.n_k_heads], qk[f.n_k_heads:]
    rep = f.n_v_heads // f.n_k_heads
    qv = np.tile(q, (rep, 1)) * (1.0 / np.sqrt(SV))
    kv = np.tile(k, (rep, 1))
    S = np.zeros((f.n_v_heads, SV, SV), np.float64)
    d = (v - np.einsum("hij,hi->hj", S, kv)) * beta[:, None]
    S = S + kv[:, :, None] * d[:, None, :]
    o = np.einsum("hij,hi->hj", S, qv).reshape(-1)
    st("attn_output-0", o)
    w = f.W(p + "ssm_norm.weight")
    n0 = rmsnorm(o.reshape(f.n_v_heads, SV), np.ones_like(w), f.eps).reshape(-1)
    st("norm-0 (rms o)", n0)
    n56 = (n0.reshape(f.n_v_heads, SV) * w[None, :]).reshape(-1)
    st("node_56 (rms*w)", n56)
    n57 = silu(z) * n56
    st("final_output-0 (pre-perm)", n57)

    # ---- the officially applied [128,16,3] -> [128,3,16] re-order -------------
    perm = np.array([3 * (h % 16) + (h // 16) for h in range(48)])
    oh = n57.reshape(48, SV)
    hp = np.empty_like(oh)
    for h in range(48):
        hp[3 * (h % 16) + (h // 16)] = oh[h]
    if args.no_perm:
        hp = oh.copy()
    n57p = hp.reshape(-1)
    n63 = n57p * f.had_signs[6144]
    st("node_63", n63)
    n65 = (n63.reshape(-1, 6, 1024) @ f.H.T).reshape(-1)
    st("node_65", n65)
    out = f.W(p + "ssm_out.weight") @ n65
    st("linear_attn_out-0", out)
    res = out + x
    st("attn_residual-0", res)
    st("post norm-0", rmsnorm(res, np.ones(5120, np.float32), f.eps))
    xb2 = rmsnorm(res, f.W(p + "post_attention_norm.weight"), f.eps)
    up, gate = f.lin(p + "ffn_up.weight", xb2), f.lin(p + "ffn_gate.weight", xb2)
    st("ffn_up-0", up)
    sw = silu(gate) * up
    st("ffn_swiglu-0", sw)
    n79 = sw * f.had_signs[17408]
    st("node_79", n79)
    n81 = (n79.reshape(-1, 17, 1024) @ f.H.T).reshape(-1)
    st("node_81", n81)
    fo = f.W(p + "ffn_down.weight") @ n81
    st("ffn_out-0", fo)
    st("l_out-0", res + fo)
    st("norm-1 (rms l_out)", rmsnorm(res + fo, np.ones(5120, np.float32), f.eps))
    print(f"\npermutation used: h -> 3*(h%16)+h//16  ({'OFF' if not args.no_perm else 'DISABLED'})")
    print("official l2(l_out-0) = 14.27, l2(attn_residual-0) = 8.31")
    return 0


if __name__ == "__main__":
    sys.exit(main())
