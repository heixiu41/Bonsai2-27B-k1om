# GDN (Gated DeltaNet) block-0 divergence — root cause

Ground truth: `D:\bonsai\dump_single.txt` (official fork, single token "The" = id 760).
Official source: `D:\llama-prism-src\llama.cpp-prism\`.

All numbers below are reproducible with the scripts listed at the end.

## Summary

Three separate defects in `layer_ssm` (`scripts/bonsai_forward.py`) and two of them
also in `block_gdn` (`src/bonsai/model.c`). All three are in the **state update /
state read-back** step and in the **hand-off to the `ssm_out` projection**; the gated
per-head RMSNorm, the `silu(z)` gate, the sign vector and the block Hadamard were
correct as written.

| # | defect | where | official behaviour |
|---|---|---|---|
| 1 | state read along the VALUE axis (`S@k`, `S@q`) while it is written key-major | numpy `layer_ssm` only (the C port is transposed-consistent and therefore correct here) | `ops.cpp:11082-11099` reads along the KEY axis: `delta[j] = (v[j] - sum_i S[i][j]k[i])*beta`, `S[i][j] += k[i]delta[j]`, `o[j] = (1/sqrt(128)) sum_i S[i][j]q[i]` |
| 2 | v head h fed by k/q head `h / 3` | both | head `h % 16` (`ops.cpp:11028` `iq1 = iv1 % neq1`; `ggml_repeat` tiles modulo) |
| 3 | missing `prism.hadamard.gdn_v_grouped` head re-order before `ssm_out` | both | `llama-model.cpp:2123-2132` + `llama-graph.cpp:1561-1568`: `.ssm_out.` activations are reshaped `[128,16,3] -> permute -> [128,3,16]` **before** the signs/Hadamard, i.e. `h' = 3*(h%16) + h/16` |

Effect at block 0 (token 760): `l2(l_out-0)` was **85.4** before, **14.240** after;
the official value is **14.270**.

## Proof 1 — the orientation, from the dump alone (no reconstruction)

For a single token the state starts at 0, so it is rank-1: `S = outer(k, delta)`,
`delta = beta*v`. `o` must therefore be parallel to *either* `v` (key-axis read,
official) or `k` (value-axis read, current numpy). Rows of `attn_output-0{128,48}` are
value heads, columns are head dims 0,1,2 / 125,126,127:

```
head 1   v = [ 0.0198  0.0183 -0.0203 ... -0.0006  0.0192 -0.0017]
         o = [ 0.0001  0.0001 -0.0001 ... -0.0000  0.0001 -0.0000]
         k = [ 0.0341 -0.1871  0.0479 ... -0.0190  0.0502 -0.0236]
         o/v = 5.05e-3 5.46e-3 4.93e-3 ... ~0 5.21e-3     <- constant, signs agree
         o/k = 2.9e-3 -5.3e-4 -2.1e-3 ...                 <- signs flip
head 45  v = [ 0.0372  0.0210 -0.0032 ... 0.0204  0.0243  0.1098]
         o = [ 0.0003  0.0002 -0.0000 ... 0.0002  0.0002  0.0009]
         o/v = 8.1e-3 9.5e-3 ... 9.8e-3 8.2e-3 8.2e-3     <- constant (0.0009/0.1098 is exact)
head 46  o/v ~ 4.5e-3 constant; o/k flips sign
```
`new_state-0{128,128,48,1}` confirms the layout: its rows are mutually parallel
(head 1 rows 0/1/2 ratios 0.83, -1.00) and each row vector is proportional to
`k_head1` (row/k ratios -5.17 vs -5.49, 1.33 vs 1.40, 1.33 vs 1.47, -0.67 vs -0.69),
i.e. the within-row axis is the KEY axis and the row-to-row axis is the VALUE axis.

## Proof 2 — the full block, all stages, sums + derived RMS

`common_debug_print_tensor`'s `sum` is the plain sum of every element of the tensor
(`common/debug.cpp:79-91`), so it is an exact statistic. `scripts/_dbg_perm.py`
(log `scripts/_dbg_perm.log`) with all three fixes:

```
tensor                   official sum      mine         ratio
attn_output-0               2.820900      2.819988      0.9997
new_state-0               124.011360    123.920581      0.9993
norm-0 (RMS of o)         515.195801    515.088842      0.9998
node_56 (RMS*w)           428.109375    428.021319      0.9998
final_output-0            -11.738554    -11.731603      0.9994   (first3/last3 identical)
node_63                    -2.755580     -2.757816      1.0008
node_65                     0.183359      0.183066      0.9984
linear_attn_out-0           6.655458      6.664997      1.0014
attn_residual-0             7.265354      7.274896      1.0013
post norm-0                62.510811     62.623624      1.0018
ffn_up-0                   -4.755027     -4.807310      1.0110
ffn_swiglu-0                0.708571      0.702457      0.9914
ffn_out-0                   2.998708      2.980901      0.9941
l_out-0                    10.264053     10.255797      0.9992
```

Two *independent* checks that do not use the sums: `RMS_NORM(l_out-0)` and
`RMS_NORM(attn_residual-0)` are unweighted RMS norms, so the dump's ratio
`norm_j / x_j` recovers `1/rms(x)` exactly (5 components agree to 0.2% each):

```
rms(attn_residual-0): official 8.310  mine 8.312   (0.02%)
rms(l_out-0):         official 14.270 mine 14.240  (0.2%)
```

## Proof 3 — end-to-end, all 64 layers

`RMS_NORM(l_out-N)` gives the official residual norm for every layer
(`scripts/bonsai_official_norms.py`). With the fixes, every *recurrent* layer sits on
that curve; the remaining error is introduced only by the full-attention layers
(`i % 4 == 3`), which are a different, separately-unverified component:

```
layer  type   mine      official   mine/official
  0    ssm     14.240    14.271     0.998
  1    ssm     20.930    20.917     1.001
  2    ssm     28.300    28.287     1.000
  3    attn   104.661    34.370     3.045   <-- full attention, not GDN
  4    ssm    104.859    37.607     2.788   (inherits layer 3's error, no further growth)
  5    ssm    105.655    39.491     2.676
  6    ssm    103.333    45.341     2.279
  7    attn   128.075    47.607     2.690
```
Before the fixes layer 0 was 85.4 and the residual grew without bound.

## Fix

`scripts/bonsai_forward.py::layer_ssm` (applied):

```python
qv = np.tile(q, (rep, 1)) * (1.0 / np.sqrt(self.state))   # was np.repeat -> h//3
kv = np.tile(k, (rep, 1))
S  = self.ssm_state[il] * np.exp(g)[:, None, None]        # S[i][j]: i = key, j = value
sk = np.einsum("hij,hi->hj", S, kv)                       # was "hij,hj->hi"
d  = (v - sk) * beta[:, None]
S  = S + kv[:, :, None] * d[:, None, :]
o  = np.einsum("hij,hi->hj", S, qv).reshape(-1)           # was "hij,hj->hi"
...
y = y.reshape(rep, self.n_k_heads, self.state).transpose(1, 0, 2).reshape(-1)
return self.lin(p + "ssm_out.weight", y)                  # gdn_v_grouped re-order
```

`src/bonsai/model.c::block_gdn` (applied, the recurrence already used the correct
transposed-consistent form): `hk = h % kh`, and the `attn_out` loop now writes
`attn_out[hp*st+d] = silu(z[h*st+d]) * o[h*st+d]` with `h = kh*(hp % rep) + hp/rep`.

Note for the plan doc: the official block-0 targets quoted there (0.114 / 1.616 / 2.531)
come from extrapolating 6 printed samples by `sqrt(n/6)`, which is invalid for these
heavy-tailed vectors. The true values are 14.27 (`l_out-0`) and 8.31 (`attn_residual-0`),
and the final hidden state is `l2(l_out-63) = 561.4`, not ~5.
