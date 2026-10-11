# Audit: C `src/bonsai/model.c` vs Python oracle `scripts/bonsai_forward.py`

> ## CORRECTION (2026-10-09, measured) — DIFFERENCE 1 below is WITHDRAWN
>
> "The GDN recurrence contracts the wrong state axis" is **not a defect**. The C
> state buffer holds the **transpose** of the oracle's `S`, so the two loop
> shapes are the *same operator in transposed layouts*, not two different
> operators. Measured with `D:\axistest\axis_h5.c` (verbatim transcription of
> both loops) plus `D:\axistest\axis_h5.py`:
>
> * `max|loop_old(S^T) - loop_new(S)| = 0.000e+00` — bit-identical;
> * over 5 tokens from the all-zero state that `bonsai_reset()` installs (the
>   only initial condition the program ever uses): `max|o_old - o_new| = 0.0` at
>   **every** token, 0/40 floats differ;
> * end-to-end: linking the pre-change `D:\zigtest\model.o` (`.text` SHA256
>   `4C0BD52C…`, bit-identical to a rebuild of the old loop) reproduces
>   `D:\zigtest\f.txt` **byte for byte** — same `my ||x|| = 1077.1537`, same
>   `rank … = 37542`, same file SHA256 `5794ECE1…`.
>
> The "max|o_old - o_new| = 2.12" in `scripts/_axis_ab.py` exists only because
> that script feeds the *same* buffer to both loops, which is not the same
> state. The script's own last two lines already show the resolution:
> `loop_old: vs oracle(S.T) max|do|=0.000e+00`, `loop_new: vs oracle(S) max|do|=0.000e+00`.
>
> The real bug found in `block_gdn` was **DIFFERENCE 4 — the L2 normalisation**
> (see the note at the end of this file): `l2norm()` was called once over the
> whole 2048-element q block and once over k, i.e. **one scale for 16 heads**,
> whereas the official `L2_NORM(view{128,32,1,1})` and the oracle's
> `l2norm(concat([q,k]), axis=-1)` both use **one scale per head**. Fixing it
> moved `l2(l_out-0)` from 38.55 to 14.2403 (official 14.2548) and
> `l2(l_out-63)` from 1072.52 to 862.07 (official single-token 561.36).

Scope read: `scripts/bonsai_forward.py` (methods `layer_ssm`, `layer_attn`, `layer`,
helpers `rmsnorm`, `silu`, `sigmoid`, `softplus`, `l2norm`, `rope_multi`),
`src/bonsai/model.c` (`block_gdn`, `block_attn`, `block_ffn`, `bonsai_forward`,
`embed_row`, `mm_folded`, `matvec_f32`, `rmsnorm`, `l2norm`, `rope_apply`,
`rope_tables`, `debug_tap`), `src/bonsai/model.h`.

All numbers below were produced with numpy 2.5.1 by transcribing each side's
index arithmetic literally (small analogue: 16 key heads -> 2, 48 value heads -> 6,
state 128 -> 6) and comparing.

---

## DIFFERENCE 1 — CRITICAL: the GDN recurrence contracts the wrong state axis

**C**: `src/bonsai/model.c` 534-549 (`block_gdn`)
**Python**: `scripts/bonsai_forward.py` 235-241

```c
534  for (int i = 0; i < st; ++i) {
535      float *row = Sh + (size_t)i * st;      /* row i */
536      float sk = 0.0f;
537      for (int j = 0; j < st; ++j) {
538          row[j] *= decay;
539          sk += row[j] * khv[j];             /* (A) contracts the SECOND index j */
540      }
542      const float d = bh * (vhv[i] - sk);    /* free index i is the FIRST index */
543      float o = 0.0f;
544      for (int j = 0; j < st; ++j) {
545          row[j] += khv[j] * d;              /* (B) update outer product on j */
546          o += row[j] * qhv[j];              /* (C) read contracts j again */
547      }
548      oh[i] = o * qscale;                    /* (D) output lands on the FIRST index */
549  }
```

```python
235  S  = self.ssm_state[il] * np.exp(g)[:, None, None]
236  sk = np.einsum("hij,hi->hj", S, kv)        # contracts the FIRST index i
237  d  = (v - sk) * beta[:, None]
238  S  = S + kv[:, :, None] * d[:, None, :]    # S[i][j] += k[i]*d[j]
239  o  = np.einsum("hij,hi->hj", S, qv)        # contracts the FIRST index i
240  self.ssm_state[il] = S
```

* `einsum("hij,hi->hj", S, kv)` sums over `i` (the **first** state axis) and the
  free/output index is `j` (the second). Verified directly: for
  `S=[[1,2],[3,4]]`, `k=[10,100]` the einsum yields `[310,420] = S.T@k`, i.e. the
  first axis is contracted.
* C's `sk` (539) sums over `j` — the **second** state axis — while `d` (542) and the
  output (548) are indexed by `i`, the first axis.
* The in-place update (545) is **not** the defect and does not need to change: the
  oracle also reads the updated state (`S` at 238 then used at 239). The defect is
  purely that the contraction axis disagrees with the axis the update/readout use.

**Consequence (measured).** With identical inputs and a persistent state:

```
C as written vs oracle : max|diff| = 3.2968   relative error  95.2 %
oracle ||o[head0]||    = 4.878     C ||o[head0]|| = 6.349
cosine(oracle, C)      = 0.675
```

End-to-end compact GDN block (conv + shared L2 + recurrence + per-head RMSNorm +
`gdn_v_grouped` permutation + output projection):

```
block output max|diff| = 0.9470 vs |oracle|max = 0.9208   (~102 % relative)
```

Diagnostic: the C recurrence applied to `S0^T` reproduces the oracle **exactly**
(`max|diff| = 0.0`). So C is running a correct recurrence on a state laid out the
wrong way round — every GDN layer output direction is wrong at ~2/3–1.0 relative
magnitude, on all 48 GDN layers of the 64.

**Minimal fix** (keeps the existing `float *S` buffer layout, keeps the update
direction, changes only which axis is contracted/read):

```c
for (int i = 0; i < st; ++i) {                 /* decay once, up front */
    float *row = Sh + (size_t)i * st;
    for (int j = 0; j < st; ++j) row[j] *= decay;
}
for (int i = 0; i < st; ++i) {                 /* i = VALUE index (column) */
    float sk = 0.0f;
    for (int a = 0; a < st; ++a) sk += Sh[(size_t)a * st + i] * khv[a];   /* sum over FIRST index */
    const float d = bh * (vhv[i] - sk);
    for (int a = 0; a < st; ++a) Sh[(size_t)a * st + i] += khv[a] * d;
    float o = 0.0f;
    for (int a = 0; a < st; ++a) o += Sh[(size_t)a * st + i] * qhv[a];
    oh[i] = o * qscale;
}
```

Verified against the oracle: **output max|diff| = 0.000e+00, state max|diff| = 0.000e+00.**

---

## DIFFERENCE 2 — `block_attn` reuses the query buffer as the attention output buffer

**C**: `src/bonsai/model.c` 654-662 (write) and 668 (read)
**Python**: `scripts/bonsai_forward.py` 291-293

```c
654  float *oh = s->q_full + (size_t)h * hd;   /* reuse q as output */
655  for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
656  for (int p = 0; p < nt; ++p) { ... oh[d] += w * vp[d]; }
...
668      s->attn_out[i] = s->q_full[i] * f_sigmoid(s->qg[qdim + i]);
```

* Python keeps `q` and the attention result `attn` in separate arrays
  (`attn = np.einsum("ht,thd->hd", w, vv)` at 291, gated at 293).
* C overwrites `q_full[h*hd .. h*hd+hd)` with head `h`'s attention output.
  I traced this: head `h` reads its own slot only at iteration `h`, and slot `h`
  is first written at iteration `h`, so there is **no cross-head corruption** —
  the loop is order-safe. The gate read at 668 reads the `gate` half of `qg`
  (616-667), not `q_full`, and is correct.
* Therefore this is currently **benign but fragile** (it survives only because the
  write index equals the read index). It is listed so it can be ruled out; unlike
  Difference 1 it does **not** change the result.

Fix (to make it robust): accumulate into a distinct buffer, e.g.
`s->attn_out + h*hd`, gate into a second buffer, and use `s->q_full` read-only.

---

## DIFFERENCE 3 — `l2norm` omits the eps floor (minor)

**C**: `src/bonsai/model.c` 47-52
**Python**: `scripts/bonsai_forward.py` 56-58

```c
const float inv = (ss > 0.0f) ? (1.0f / sqrtf((float)ss)) : 0.0f;
```
```python
n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
return x / np.maximum(n, eps)      # eps = 1e-6
```

C divides by `sqrt(ss)`; Python divides by `max(sqrt(ss), 1e-6)`. Different only
when `ss < 1e-12`. Conv activations here are O(1)-O(10) per element, so this is
worth < 1e-12 relative and cannot explain a 2x error. **Not a contributor.**

---

## AREAS CHECKED AND FOUND EQUIVALENT (ruled out)

| Area | C lines | Python lines | Result |
|---|---|---|---|
| State axis convention in the *update* (545 vs 238) | 545 | 238 | equivalent — same rank-1 form; not the bug |
| `q` scaling `1/sqrt(st)` applied at the same point | 519, 548 | 224, 239 | equivalent (constant scales commute with the contraction) |
| v-head -> k/q-head mapping (`h % kh`) | 525 | 224-225 | equivalent (`np.tile(q,(rep,1))` gives `h % kh`) |
| Per-head RMSNorm over `st`=128, weight reused per head | 555-558 | 243-244 | equivalent; axis is the head dim, not the 6144 vector |
| RMSNorm epsilon source & reduction axis | 41-43, 442, 557 | 37-40, 196, 244 | same eps (`m->rms_eps`), same `mean(x*x)` over the reduced axis |
| `gdn_v_grouped` head permutation `h' = 3*(h%16)+h/16` | 566-571 | 255-256 | **exact match** (verified: `max|diff| = 0.0`), applied to the gated buffer **before** the matmul, direction correct |
| Conv indexing `w[c*4+k]` and history k-major | 483-497 | 207-210 | equivalent (verified numerically) |
| RoPE pairing (i, i+half) and theta | 164-189 | 61-78 | equivalent |
| `rope_tables` / `rope_apply` angle formula | 181-188 | 70-72 | equivalent |
| `mm_folded` output dim `dims[1]`, signs-then-Hadamard | 104-138 | 183-191 | equivalent |
| `matvec_f32` row-major `w[o*n+i]` | 148-155 | 201-203 | equivalent |
| `embed_row` H-then-signs inverse transform | 711-726 | 335-343 | equivalent |
| `block_ffn` norm/gate/up/silu/down/residual order | 680-702 | 300-302 | equivalent |
| `block_attn` q/k per-head norm -> RoPE -> GQA -> softmax -> gate | 590-668 | 259-293 | equivalent (see Diff. 2 for the aliasing caveat) |
| `bonsai_forward` layer-type selection + residual + final norm | 728-757 | 295-302, 346-351 | equivalent |
| `debug_tap` | 400-408 | n/a | no-op unless `debug` set |

Two non-divergences worth recording: C accumulates the recurrence in `float`
(535-547) where numpy uses pairwise summation, and the LM weights are consumed in
PQ2_0 (2.125 bpw) form; both are format/precision choices shared with the C design,
not porting divergences.

---

## RANKING — what can explain 1077 vs 561 after 64 layers

> **This ranking is WRONG and is kept only as a record.** Item 1 was withdrawn
> (see the correction at the top); the actual cause is item 3's neighbour, the
> **per-head vs per-block L2 normalisation**. Measured after the fix: `l2(l_out-0)`
> = 14.2403 (official 14.2548), `l2(l_out-63)` = 862.07 for token 760 (official
> 561.36); the remaining factor 1.535 comes entirely from the 16 full-attention
> blocks, whose first appearance (layer 3) is off by 3.045x in the C **and in the
> numpy oracle alike** (`docs/GDN_ROOTCAUSE.md` lines 89-98). The GDN/ssm path now
> matches the oracle's post-fix curve to three decimals at every layer
> (14.240 / 20.930 / 28.300 / 104.661 / 104.859 / 105.655 / 103.333 / 128.075).

1. **Difference 1 (GDN contraction axis).** The only semantic divergence found.
   48 of 64 layers are GDN, and each of their outputs is wrong with ~95-100 %
   relative per-head error (`cosine = 0.675`, i.e. ~48 deg off), so each GDN layer
   injects a contribution of the right magnitude but the wrong *direction* into the
   residual stream. Over 64 layers that systematically inflates the accumulated
   norm; a final-factor of ~1.9 is consistent with a direction error of this size.
   **This is the root cause.** Fix it first and re-measure; no other divergence
   found can move the norm at all.
2. **Difference 2 (attention query/output aliasing).** Currently benign, so its
   contribution is ~0 — but it becomes an active, per-head data-destroying bug the
   moment the loop is reordered or vectorised. Fix it defensively.
3. **Difference 3 (l2norm eps floor).** < 1e-12 relative. Cannot contribute.

The known hints to re-verify: "contracting the SECOND axis ... is a different
recurrence" — confirmed, and it is the bug; "C scales the same quantity at the same
point" — confirmed equivalent; "same epsilon source and reduction axis" — confirmed;
"permutes the RIGHT buffer ... inverse direction" — confirmed correct, permutation is
exact; "conv weight indexing `w[c*4+k]` and history shift" — confirmed equivalent.
