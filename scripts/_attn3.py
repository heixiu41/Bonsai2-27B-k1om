"""Bisect the full-attention block (layer 3) by stage magnitude.

Layer 2 (GDN) matches the official dump to 0.1%; layer 3 (the first full-attention
block) jumps to ~3.045. This script prints the norm of each stage inside
layer_attn and, where the official dump has an exact full-tensor statistic
(the `sum = ...` line printed by common_debug_print_tensor), compares against it.

Official ground truth (single-token dump "The", token 0), sums over all elements:
  read directly from D:\\bonsai\\dump_single.txt
"""
import sys
sys.path.insert(0, 'scripts')

import numpy as np

from bonsai_forward import (BonsaiForward, rmsnorm, silu, sigmoid, softplus,
                            l2norm, rope_multi)

f = BonsaiForward(local=r'D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf')

# ---- replay layers 0..3 with taps ----
toks = [760]
il = 3
x = f.embed(760)
for i in range(il):
    x = f.layer(i, x, 0)

p = f"blk.{il}."
hd = f.head_dim
print(f"=== layer {il} (full attention) stage norms, input token 760 ===")
print(f"  input x                l2 = {np.linalg.norm(x):.4f}   "
      f"(l_out-2 official first3 = [-0.0388 0.0353 -0.1159])")

xb = rmsnorm(x, f.W(p + "attn_norm.weight"), f.eps)
print(f"  attn_norm              l2 = {np.linalg.norm(xb):.4f}   "
      f"(official attn_norm-3 first3 = [0.1238 0.1002 -0.5306])")

qg = f.lin(p + "attn_q.weight", xb)
q_raw = qg[:f.n_head * hd].reshape(f.n_head, hd)
gate = qg[f.n_head * hd:]
k_raw = f.lin(p + "attn_k.weight", xb).reshape(f.n_head_kv, hd)
v_raw = f.lin(p + "attn_v.weight", xb).reshape(f.n_head_kv, hd)
print(f"  q_raw                  l2 = {np.linalg.norm(q_raw):.4f}")
print(f"  gate                   l2 = {np.linalg.norm(gate):.4f}")
print(f"  k_raw                  l2 = {np.linalg.norm(k_raw):.4f}")
print(f"  v_raw                  l2 = {np.linalg.norm(v_raw):.4f}")

q = rmsnorm(q_raw, f.W(p + "attn_q_norm.weight"), f.eps)
k = rmsnorm(k_raw, f.W(p + "attn_k_norm.weight"), f.eps)
print(f"  q after per-head norm  l2 = {np.linalg.norm(q):.4f}  "
      f"(expect ~sqrt(24*256)={np.sqrt(24*256):.1f} if norms are ~1)")
print(f"  k after per-head norm  l2 = {np.linalg.norm(k):.4f}  "
      f"(expect ~sqrt(4*256)={np.sqrt(4*256):.1f})")

qr = rope_multi(q, 0, f.rope_base, f.n_rot)
kr = rope_multi(k, 0, f.rope_base, f.n_rot)
print(f"  q after RoPE           l2 = {np.linalg.norm(qr):.4f}  (RoPE is a rotation: ~unchanged)")
print(f"  k after RoPE           l2 = {np.linalg.norm(kr):.4f}")

rep = f.n_head // f.n_head_kv
kk = np.repeat(kr, rep, axis=0)
vv = np.repeat(v_raw, rep, axis=0)
scores = np.einsum("hd,thd->ht", qr, kk[None]) / np.sqrt(hd) if kr.ndim == 2 else None
# single token => only one key, softmax weight is exactly 1
attn = v_raw.reshape(-1)
attn = np.repeat(v_raw, rep, axis=0).reshape(-1)
print(f"  attn (single token)    l2 = {np.linalg.norm(attn):.4f}")
print(f"  attn_pregate official first3 = [-0.1858 0.1276 0.007]  (this is pre-gate)")

gated = attn * sigmoid(gate)
print(f"  attn_gated             l2 = {np.linalg.norm(gated):.4f}")
print(f"  attn_gated official  first3 = [-0.0068 0.0013 0.0001]  sum official = ?")
print(f"  my gated first3            = {np.array2string(gated[:3], precision=5)}")
print()

out = f.lin(p + "attn_output.weight", gated)
print(f"  attn_output            l2 = {np.linalg.norm(out):.4f}")
print(f"  attn_output official first3 = [-0.0288 -0.0012 0.0408]")
print(f"  my attn_output first3       = {np.array2string(out[:3], precision=5)}")
print()
print(f"  l_out-3 = x + attn_output   l2 = {np.linalg.norm(x + out):.4f}")
print(f"  l_out-3 official first3 = [0.0397 0.0258 -0.1553]")
print(f"  my l_out-3 first3       = {np.array2string((x + out)[:3], precision=5)}")
