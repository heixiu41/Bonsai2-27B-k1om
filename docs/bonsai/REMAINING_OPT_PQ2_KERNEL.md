# Remaining optimization opportunities — pq2_0.h / gguf.c (static analysis, read-only)

Scope: `src/pq2_0.h` (F3 kernel, B2 scale builder, xp build, Hadamard) + `src/gguf.c`
(loader). Call-site facts taken from `src/model.c` (mm_stage/mm_run). All cycle/time
figures are arithmetic on existing card measurements; **no new card runs**.

## Answers to the seven questions (evidence first)

**1. F3 block-loop op count.** Emitted count for the V12-era loop is measured, not
estimated: 30 in-loop instructions per 128 weights (pq2v16 disassembly,
`scripts/pq2v12_card_evidence.txt:289-294`). F3 swaps V12's per-block scalar fp16
decode for a `drow[]` load, so F3 ≈ 30–33. Intrinsic-level recount of
`pq2_0.h:479-499`: set1(1) + 2 mul + 2×LOAD16(4) + 2 srli + 8 permutevar + 8 FMA +
8 xp loads + 1 add = **34 vector ops** before compiler folding (some xp loads fold
into FMA memory operands; evidence shows 6 standalone vmovaps + 1 vprefetch1).
- *Hoist the 2 vmulps out of the loop?* **Impossible** — `vd = set1_ps(*dp)` and
  `++dp` (pq2_0.h:480,498): the LUTs change per block. Confirmed as the prompt
  suspected.
- *Eliminate `s = add(s,t)`?* No. One partial-sum vector needs one fold per block;
  two alternating accumulators still cost 1 add/block plus an extra final add. The
  s-chain (1 add per ~275-cycle block) is not the critical path, and F3's single
  accumulator already beat V12's four (140.4 vs 103.9 G w/s, measured).
- *Pointer-increment vs indexed addressing:* runs on the scalar pipe, dual-issues
  under vector work — free either way. Leave it.

**2. drow build cost — much bigger than "small".** Measured single-thread
(`scripts/pq2aid_card_evidence.txt:292,308,316`): build alone = **204 ps/weight**;
F3 kernel-only = 2400 M w/s, F3+B2 = 1669 M w/s → the build is **~30% of row cost
at 1T**. Per row at n=5120: 2 gathers + an 8-block **scalar tail** (n=5120 → 40
blocks = 2×16+8; n=17408 → 136 = 8×16+8 — note: 8 gathers + scalar tail, not "9
gathers"). One gather group costs ~550 cycles (vpgatherdd is microcoded, stride-34
touches 9 cache lines). At 244t the fraction is unmeasured; if it scales like the
kernel it is 15–30% of the 190 ms matvec = **28–57 ms/token**. The scales are
token-invariant, so the entire build is redundant work repeated every token.

**3. xp build — the prompt's "~0.05 ms × 258 ≈ 13 ms" is optimistic on both
factors.** `pq2_build_xp` runs inside `pq2_0_matvec_knc` (pq2_0.h:463), i.e. **per
mm_run, not per staging: 401×/token** (273 calls at n=5120, 64 at n=6144, 64 at
n=17408), serial on the master while 243 threads spin. Cost ≈ 273×0.05 + 64×0.06 +
64×0.17 ≈ **28 ms/token** (0.041–0.056 ms @ n=5120 measured, HANDOVER.md:189,
scaled linearly). 143 of the 401 builds (z after qkv; k,v after qg; up after gate)
re-permute the *identical* fold_x — pure waste, ~7.2 ms. Gather-based build is
correctly rejected (45-cycle microcoded gather vs ~7 permute/blend ops per 16
elements). But the permutation `xp[(b*2+h)*64+f*16+l] = x[b*128+64h+4l+f]` is a
fixed 4-source lane pattern: out chunk (h,f) lane l = x-vector l/4, lane 4(l%4)+f —
buildable with 4 permutevar + 3 mask_blend_epi32 per 16 floats ≈ 72 ops/128
elements vs ~12 cycles/element scalar (~10× cheaper), or simply moved inside the
existing omp region (blocks are independent; zero extra fork).

**4. Thread partitioning — small tails.** schedule(static) idle-row fraction =
(244·⌈d/244⌉−d)/(244·⌈d/244⌉): d=5120 → 0.08%, 10240 → 0.08%, 12288 → 1.3%,
17408 → 0.9%, 6144 → 3.3%, 2048 (attn k/v) → 7.3%. Weighted by matvec time:
**~1.4 ms/token total**. Collapsing same-n groups (gate+up → 34816 rows; q/k/v →
16384; qkv+z → 16384; all share n=5120) roughly halves the non-down tails and
drops 143 omp forks → ~0.5–1 ms/token. Real but marginal.

**5. Hadamard staging passes.** Per staging at n=5120: memcpy (1R+1W) → **scalar**
sign loop (1R+1W, model.c:182-183) → fwht (10 levels, 10R+10W) → vecscale
(1R+1W) = 13 array passes, then each mm_run adds the xp pass. Two redundancies:
(a) `hadamard_forward`'s *vectorized* sign path (pq2_0.h:751-759,
vecmul_inplace_mic) is **dead code for the model** — mm_stage passes `signs=NULL`
and does its own scalar loop; sign+copy measured at 11 ms/token back when there
were 401 calls (K1OM_TIME_BUDGET.md:105), so ~7 ms/token now. (b) vecscale is a
separable pass that fuses into the last butterfly level with *identical* rounding:
fl((u+v)·norm) either way. Note the vector path also requires signs_* to become
64-byte aligned (model.c:463-465 uses malloc — only 16 B).

**6. h<16 scalar butterfly levels.** Per 1024-block: levels h=1,2,4,8 = 2048 scalar
butterflies ≈ 12k instructions ≈ **91% of FWHT instruction count** (the 6 vector
levels ≈ 1150). Vectorizable in-register, bit-identically: per 16-float chunk,
load once, then per level `p = permutevar(xor_h_index, v); s = v+p; d = v−p;
v = mask_blend_epi32(h_mask, s, d)` (all four ops are on the proven-linkable
list), 4 levels, store once ≈ 18 ops/chunk vs ~192 scalar — FWHT instruction count
drops ~5–6×. Each lane still performs exactly one add or sub on the same operand
pair, so results are bit-identical (same argument as the h≥16 vectorization).

**7. Loader.** Nothing egregious: one 7.2 GB fread, one-pass parse, ~110 MB wcache
dequant. The ~15 s startup is fread-bound on a 1.2 GHz in-order core. Parallelizing
wcache_fill would shave <1 s of startup. Not worth touching.

## Ranked table

| rank | what & where | mechanism | est. saving ms/token (arithmetic) | correctness risk | verification plan |
|---|---|---|---|---|---|
| 1 | Build xp once per **staging** and build it in parallel or vectorized — pq2_0.h:392-400,463; scratch field fold_xp (already sketched in docs/bonsai/V12_INTEGRATION_PLAN.md:105) | kills 143/401 redundant builds (7.2 ms) + removes ~21 ms serial-master time: 273×0.05+64×0.06+64×0.17 = 28.4 ms → ~1–3 ms | **~25** | none: pure permutation, bit-identical; host path untouched (ref reads fold_x) | diff xp vs scalar build at n=5120/6144/17408 on random x; then top-1=11751, rank 0, l_out-63 must be **bit-identical** |
| 2 | Precompute all fp32 block scales at load; kernel indexes a contiguous array, `pq2_build_drow` deleted — pq2_0.h:410-441,471-473 | build measured 204 ps/w and ~30% of row cost at 1T (1669 vs 2400 M w/s); at 244t band 15–30% of 190 ms matvec | **28–57** (point ~35) | none numerically (B2 already proven bit-exact vs the scalar converter on all 65536 halves; kernel math unchanged → **bit-identical**). Cost: +848 MB resident (or +424 MB keeping fp16 contiguous and converting inline) — check 15.5 GB budget at ctx=8192 (≈13.4→14.2 GB) | **first** a pq2ab-style probe at 244t with drow prebuilt outside the timed region to size the win; then bit-compare precomputed array vs pq2_build_drow on every tensor; end-to-end must be bit-identical |
| 3 | Rescale after the block sum: drop the 2 LUT vmulps + final add; `s = fmadd(vd, t, s)` with unscaled LUTs — pq2_0.h:481-482,495 | removes the broadcast→2-mul prologue from the per-block dependency chain; measured class: breaking the d-chain was worth 1.38–1.40× at 1T (pq2v13 Q1/Q5, evidence:165,170); F3+B2 already captured the decode half | **0–19** (0 if the 244t ceiling is pure memory latency; ~19 if 10% of matvec) | ~1e-7-class drift, **not** bit-identical: lut·x is exact (lut∈{-1,0,1,2}), one d·t rounding per block replaces per-element (lut·d)·x rounding — same drift class as the accepted V12→F3 change | same-binary A/B at 1T and 244t (pq2ab harness exists); gates: top-1 11751 MATCH + rank 0 + l_out-63 within current margin (8.7% slack is the binding constraint) |
| 4 | Route the sign multiply through the existing vector path: mm_stage passes `signs` to hadamard_forward instead of its own scalar loop; allocate signs_* with aligned_f32 — model.c:182-183,463-465; pq2_0.h:751-759 | sign+copy measured 11 ms/token at 401 stagings → ~7 ms now; vector mul ≈ 0.5 ms; optional one-pass copy×sign fusion (same single mul per element) | **4–5** | none: one multiply per element either order → bit-identical | standard gates; expect bit-identical end-to-end |
| 5 | Vectorize FWHT levels h=1,2,4,8 in-register (permutevar + add/sub + mask_blend_epi32) — pq2_0.h:638-647 | scalar levels are ~91% of FWHT instructions (12k of 13.2k per 1024-block); ~5–6× fewer instructions | **1.5–3** (Hadamard's current post-vectorization cost is not separately instrumented — assumption band) | none: identical per-element add/sub, exact data movement | bit-compare vs scalar fwht_inplace at block=1024 incl. ±0/subnormal/adversarial patterns; then standard gates, bit-identical expected |
| 6 | Fuse the 1/√n vecscale into the last butterfly level — pq2_0.h:648-657,766 | removes one full R+W pass over fold_x per staging (258×/token) | 0.3–1 | none: fl((u+v)·norm) computed identically | standard gates; bit-identical expected |
| 7 | Collapse same-n matvec groups into one parallel region (virtual row index) | idle-row tails total ~1.4 ms/token (0.08–7.3% per shape; 2048-row k/v worst at 7.3% of 2.4 ms); collapsing halves that + drops 143 forks | 0.5–1 | none: same per-row computation | standard gates; bit-identical expected |

## Considered and rejected

- **Hoisting the 2 LUT vmulps out of the block loop** — impossible: `vd` is
  per-block (pq2_0.h:480,498).
- **Eliminating the per-block `s=add(s,t)` / alternating accumulators** — no op is
  removed, the s-chain is not the critical path, and F3's 1-accumulator form
  already beat 4 accumulators by measurement.
- **Fixing the drow scalar tail with an overlapped final gather** (n=5120: 40 =
  2×16+8) — a gather group costs ~550 cycles vs ~200 for the 8 scalar converts;
  the "fix" *loses* ~350 cycles/row. Rejected standalone; moot if rank 2 lands.
- **Gather-based xp build** — vgatherdps is microcoded (latency 45); the permute/
  blend scheme is ~7 ops per 16 elements. Rejected by op count.
- **Two-row register blocking** (share xp loads across row pairs) — bit-identical
  and −12% ops on paper, but V15's K>1 multi-block unroll measured *worse*
  (1316–1427 vs 1466 M w/s, register pressure) and at 244t decode ops are free.
- **16-bit SWAR / maddubs / IEEE bias-trick LUT replacement** — instructions don't
  exist on k1om (KNC_INTEGER_SIMD.md) or measured slower/wrong (pq2v13 Q2:
  vpaddd ≡ vpermd; V3: 2.1× slower and numerically wrong).
- **Token batching** — measured refuted (pq2f3batch: B=2/4/8/16 = 0.91/0.99/0.97/0.83×).
- **Loader work** — 15 s startup is fread-bound; nothing egregious in gguf.c.

## Honest uncertainty

- **The ranking's fork in the road**: ranks 2–3 attack the kernel's 190 ms and are
  worth 0–70 ms depending on what actually sets the 244t ceiling (prologue-chain
  latency vs weight-stream memory latency — the B2 gathers also pre-touch the
  lines the block loop then reads, so removing them could hurt as well as help).
  One cheap card probe decides it: pq2ab variant with drow prebuilt outside the
  timed loop at 244t. Ranks 1/4/5/6 (~30 ms, ~10% end-to-end) attack **serial
  master time** and are independent of that question.
- The 0.05 ms xp figure is V10-era at n=5120; re-measure before sizing rank 1.
- Hadamard's current per-token cost after the block-parallel + vectorized rewrite
  is not separately instrumented; rank 5 assumes 3–6 ms/token.
- Ceiling if everything lands: 313 ms → ~210–250 ms/token (3.2 → ~4.0–4.7 tok/s).
