#!/usr/bin/env python3
"""Is scripts/bonsai_forward.py's RoPE the same transform as model.c's?

oracle  rope_multi():  x[:n_rot].reshape(..., nr, 2)  -> pairs (2i, 2i+1)   [interleaved]
model.c rope_apply():  (x[i], x[i + n_rot/2])         -> pairs (i, i + 32)  [NeoX / rotate_half]
Both document themselves as "neox-style rotate_half".

Run:  python _review/rope_check.py
"""
import numpy as np

n_rot, pos, base = 64, 3, 1e7

def oracle_rope(x):
    nr = n_rot // 2
    i = np.arange(nr, dtype=np.float64)
    ang = pos * (base ** (-2.0 * i / n_rot))
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    out = x.copy()
    v = x[..., :n_rot].reshape(*x.shape[:-1], nr, 2)
    v0, v1 = v[..., 0].copy(), v[..., 1].copy()
    out[..., :n_rot] = np.stack([v0 * cos - v1 * sin,
                                 v0 * sin + v1 * cos], axis=-1).reshape(*x.shape[:-1], n_rot)
    return out

def modelc_rope(x):
    """verbatim transcription of model.c rope_apply()/rope_tables()."""
    half = n_rot // 2
    out = x.copy()
    i = np.arange(half, dtype=np.float64)
    theta = pos * (base ** (-2.0 * i / n_rot))
    cos, sin = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    for i in range(half):
        a, b = x[i], x[i + half]
        out[i]        = a * cos[i] - b * sin[i]
        out[i + half] = a * sin[i] + b * cos[i]
    return out

rng = np.random.default_rng(0)
x = rng.standard_normal(n_rot).astype(np.float32)

# which element does the oracle pair with element 0?
pairs = []
for i in range(n_rot // 2):
    # find j such that oracle output[0] depends on x[i] and x[j]
    pass
v = x.reshape(n_rot // 2, 2)
print("oracle pairing (from reshape(nr,2)) :", [(2 * i, 2 * i + 1) for i in range(4)], "...")
print("model.c pairing (half-split)        :", [(i, i + n_rot // 2) for i in range(4)], "...")
o, c = oracle_rope(x), modelc_rope(x)
print("\nmax |oracle - model.c| on one head  : %.6f" % np.abs(o - c).max())
print("oracle  [:8] =", np.round(o[:8], 5))
print("model.c [:8] =", np.round(c[:8], 5))
print("\nidentical transform? ", np.allclose(o, c))
