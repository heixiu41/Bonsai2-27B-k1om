# GDN axis change (no-op) and the L2 normalisation bug — measured findings

Date: 2026-10-09. All numbers below were produced by building and running code, not by
inspecting index arithmetic. Artifacts live in `D:\axistest\` (sources + logs) and
`D:\zigtest\` (binaries + outputs).

## 1. Why the "axis fix" could not change the output (H5 holds; H1–H4 refuted)

The C `ssm_state` buffer holds the **transpose** of the numpy oracle's `S`:
`Sh[a*st + i] == S_oracle[i][a]`. Both loop shapes are the *same recurrence in
transposed layouts*, and each is self-consistent — nothing outside the loop ever reads
`ssm_state`. Since `bonsai_reset()` memsets the state to zero (a self-transpose value)
and the transpose relation is preserved by the decay/update, the two shapes agree
**bit for bit** from the only initial condition this program ever uses.

| measurement | value |
|---|---|
| `max|C loop_old(S0^T) - C loop_new(S0)|` (`axis_h5.c`) | `0.000e+00` |
| `max|C loop_new - numpy oracle(S0)|` | `0.000e+00` |
| `max|C loop_old - numpy oracle(S0^T)|` | `4.47e-08` (float32 accumulation only) |
| `max|C loop_old(S0) - C loop_new(S0)|` (same buffer — what `_axis_ab.py` measures) | `6.40e-01` |
| 5 tokens from a zero state, same inputs, each loop on its own buffer | `0.000e+00` at **every** token, **0/40** floats differ |
| pre-fix `model.o` linked into `test_forward` and run | output file SHA256 `5794ECE1FE876E663FE7FC2A603843233C335E15E3661D00227A6BA4998AA012` = `f.txt` = `f2.txt`, **byte-identical**, same `my ||x|| = 1077.1537`, same `rank = 37542` |

`scripts/_axis_ab.py` already printed the resolution in its own last two lines:
`loop_old: vs oracle(S.T) max|do|=0.000e+00`, `loop_new: vs oracle(S) max|do|=0.000e+00`.
Its "2.12 difference" comes only from feeding the *same* buffer to both loops, which is
not the same state.

Object-level evidence for what the running binary contained:

* `D:\zigtest\model.o` (pre-fix, 20:10) `.text` = `31348` bytes, SHA256 `4C0BD52C…` —
  **bit-identical** to a rebuild of the current source with the old second-index loop
  restored, and **different** from the current `model.c` object (`32532` bytes).
* `D:\zigtest\tf.exe` (20:30) `.text` = `203264` bytes, SHA256 `4D130C5E…` —
  **bit-identical** to a fresh `test_forward.c model.c gguf.c` build from the current
  source. So the shipped binary did contain the `a < st` loop (H3 refuted: the stale
  object was never linked; linking it yields a 264192-byte exe, not 265216).

Conclusion: the axis change is a pure layout re-parameterisation. Neither shape is a
bug and no output can move either way. The false "different operator … dominates the
whole stack" comment in `block_gdn` has been replaced with the measurement.

### H2 (measured): the reported number is the wrong token, but that is not the 2x

`test_forward.c` sets the tap once and then runs the 5 forwards, so `debug_tap()`
overwrites `m->debug` each time and the printed `||x||` is the **last** token
(pos 4, id 369), while the comment compares it with the single-token dump.

| | before fix | after fix | official |
|---|---|---|---|
| token 760 alone (`probe3`) | `1072.5220` | `862.0699` | `561.3632` |
| harness value = last token (pos 4) | `1077.1537` | `1028.7740` | `519.5392` (see below) |

The 5-token official dump `D:\bonsai\eval_callback_full.txt` is for exactly
`760, 6511, 314, 9338, 369`. Its layer-63 FFN is computed for a single token (the
output-token restriction), so its `l_out-63` *is* the last token:
`sum = 46.752823`, `first3 = [8.2843 -4.3284 5.6489]`, recovered `l2 = 519.5392`
(same method that gives `561.3648` from `dump_single.txt`). Ratios: `2.073` before,
`1.980` after. So the ~2x verdict holds for both readings, but the harness number is
not the one the comment claims, and it is the wrong comparison for `561.4`.

## 2. The real bug: L2 normalisation was per *block*, not per *head*

* Official: `L2_NORM(conv_output_silu-0 (view){128, 32, 1, 1})`
  → `ggml_compute_forward_l2_norm_f32` (`D:\llama-prism-src\llama.cpp-prism\ggml\src\ggml-cpu\ops.cpp:4195-4218`)
  computes **one scale per row over `ne00`**: 32 independent norms of 128
  (16 q heads + 16 k heads).
* Oracle: `scripts/bonsai_forward.py:216` `l2norm(np.concatenate([q, k], axis=0))`
  with `axis=-1` → identical to per-head normalisation
  (`max|diff| = 0.0` in `axis_h5.py`).
* Old C: `l2norm(s->q, s->conv_out, kh*st)` — **one scalar for 16 heads**, same for k.
  Real block-0 data: per-head norms spread **10.0x** (q) and **18.8x** (k);
  mis-scaling of individual heads up to **93.9%**.
* The literal "single shared L2 over q and k" (one scalar over 4096) is **also wrong**:
  `|oracle - one scalar over q+k| = 3.13e-01` on synthetic data. The fix must be
  32 row norms.

**Why it is not invisible** (a per-head scalar would cancel in the following per-head
RMSNorm — but the eps floor breaks that). Measured on real block-0 data, `probe5`:

| | old (block scalar) | fixed (per head) | official |
|---|---|---|---|
| `sum(raw DeltaNet o)` = `attn_output-0` | `0.035170` (0.0125x) | `2.819989` (**0.9997x**) | `2.820900` |
| `mean(o^2)` (mean over heads) | `6.03e-09` | `4.63e-05` | |
| `mean(o^2)/rms_eps` (`rms_eps = 1e-06`) | **`6.03e-03`** | `46.3` | |
| RMSNorm scale actually applied | `997.06` (≈`1/sqrt(eps)` = 1000) | `484.30` | |

With the old normalisation the pre-norm `o` is ~80x too small, so `mean(o^2)` falls
below `rms_eps`, RMSNorm stops normalising (same ~1000 scale for every head) and the
q/k scale error no longer cancels — even for a single token with a fresh state.

## 3. What the fix changes (single token 760, `probe3` vs `dump_single.txt`)

| layer | type | before | after | official | ratio before | ratio after |
|---|---|---|---|---|---|---|
| 0 | ssm | 38.5455 | **14.2403** | 14.2548 | 2.704 | **0.9990** |
| 1 | ssm | 44.9506 | **20.9298** | 20.9158 | 2.149 | **1.0007** |
| 2 | ssm | 44.8305 | **28.2997** | 28.2826 | 1.585 | **1.0006** |
| 3 | attn | 111.4911 | 104.6610 | 34.3610 | 3.245 | **3.0459** |
| 4 | ssm | 111.8068 | 104.8587 | 37.6028 | 2.973 | 2.789 |
| 5 | ssm | 111.6793 | 105.6552 | 39.4862 | 2.828 | 2.676 |
| 6 | ssm | 109.8516 | 103.3330 | 45.3392 | 2.423 | 2.279 |
| 7 | attn | 148.0655 | 128.0752 | 47.5969 | 3.111 | 2.691 |
| 63 | attn | 1072.5220 | **862.0699** | 561.3632 | 1.911 | **1.536** |

* The GDN prefix (layers 0–2) now matches the official dump to 0.1%.
* The whole post-fix curve equals the numpy oracle's documented post-fix curve
  (`docs/GDN_ROOTCAUSE.md:89-98`) to three decimals:
  `14.240 / 20.930 / 28.300 / 104.661 / 104.859 / 105.655 / 103.333 / 128.075`.
* The remaining error enters **only** at the full-attention blocks: the ratio is
  1.000 at the last GDN layer before the first attention block and 3.046 at that block,
  with the identical 3.045 signature in the numpy oracle. `block_attn` is the next
  target; it is not a GDN problem.
* End-to-end, the harness after the fix: `my ||x|| = 1028.7740`
  (was `1077.1537`), `rank of 11751 = 14979` (was `37542`), `top-1 = 17`
  (was `248046`).

## 4. Changes made

* `src/bonsai/model.c` — `block_gdn`: q/k are now L2-normalised **one scale per head**
  (32 norms of `st`); plus the two false comments about the "axis bug" replaced by the
  measurement above.
* `docs/_AUDIT_GDN_C_vs_PY.md` — correction header; "DIFFERENCE 1" withdrawn; the
  RANKING section annotated as wrong.

## 5. Reproduction

```
D:\zig\zig-x86_64-windows-0.17.0\zig.exe cc -O2 -std=c11 -o D:\axistest\axis_h5.exe D:\axistest\axis_h5.c
D:\axistest\axis_h5.exe > D:\axistest\h5_c.txt ; python D:\axistest\axis_h5.py
D:\zig\...\zig.exe cc -O2 -std=c11 -o D:\axistest\probe3.exe D:\axistest\probe3.c D:\axistest\gguf.c
D:\axistest\probe3.exe D:/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf 760 > D:\axistest\probe3_fix.txt
python D:\axistest\diff_layers.py D:\bonsai\dump_single.txt D:\axistest\probe3_fix.txt
```
