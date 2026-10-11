# PQ2_0 decoder: V12 -- 1.55-1.56x verified-exact, and why the vpermd premise was wrong

> Card: Intel Xeon Phi 7120P (k1om), MPSS 3.8.6, `cpu MHz 1238.094`, single
> thread, load average 0.00.
> All numbers below are card measurements; raw output is in
> `src/bonsai/pq2v12_card_evidence.txt`.
> **`pq2_0.h` was NOT modified.** Integration is left to the reader, together
> with the caveats in section 7.

---

## 1. Result

| kernel | M weights/s | ratio vs V1 | exact? |
|---|---|---|---|
| V1 (shipped `pq2_0_dequant_block_imci` + FMA) | 923-941 | 1.000 | reference |
| V10c (previous agent's verified best) | 1052-1067 | **1.14x** | exact |
| V12 (this work) | 1450-1472 | **1.55-1.58x** | exact |

V12 is **verified exact to 0 mismatches over 26 832 element-wise checks**
(64 source alignments x (128 unit vectors + row sum), plus 8 finite random
scales and 10 edge-case scales (0, -0, subnormals, max normal, negative) x 8
alignments x 129 checks, all against the shipped decoder), and is **bit-identical
to V10c** on multi-block rows.

Two runs of the same binary with the price list in *forward* and *reverse* order
agree to within 1.5%:

```
  kernel                pass1    pass2
  V1 shipped            1.000    1.000     (934.8 / 939.2 M w/s)
  V10c                  1.142    1.136
  V12 inline            1.318    1.310
  V12 helper            1.570    1.556
  V12 inline+pre        1.575    1.551
  V12 helper+pre        1.564    1.544
```

The V10c cross-check (1.14x here vs the previous agent's 1.142-1.146x) says the
harness agrees with theirs.

---

## 2. The measurement that changed the plan: there is no "expensive op class"

The project's working premise was "`vpermd` is KNC's microcoded instruction and
is 62% of the decoder". Three independent microbenchmarks
(`pq2prim.c`, `pq2prim2.c`, `pq2prim3.c`) measured the primitives directly, with
8 independent register-only chains and every result consumed into a volatile
sink, and then checked the loops in the disassembly (`srccount.py`):

```
  vpaddd / vpandd / vpord / vpsrld / vpermd / vpmulld / vmulps / vfmadd
  all measure 1625-1630 ps/op  (2.01 cycles at 1238 MHz)  -- within 1%
  a memory operand costs nothing extra (12998 vs 13018 ps per iteration)
  8 independent ops in a loop take 13017 ps = 16.1 cycles  => 2.01 cycles/op
```

So on this part **every 512-bit ALU operation costs the same**; the only
currency is the *number* of vector operations. That was then confirmed *inside
the real kernel* (`pq2v13` probe Q2): replacing all 8 `vpermd` of V12 by 8
`vpaddd`, same op count, same loop:

```
  V12                      1229.5 M w/s
  Q2  8 adds instead of 8 permd   1228.6 M w/s     (+0.0%)
  Q3  4 permd + 4 adds            1233.3 M w/s     (+0.3%)
```

**Consequences for the ideas in the brief:**

* Angle 1 ("split the LUT across bytes") and angle 3 ("use the cheaper
  restricted shuffles") are both dead for this reason: they aim to replace a
  permute with something cheaper, and nothing is cheaper. (Angle 1 is also
  impossible by construction -- `permutevar(idx,data)` returns a whole dword and
  consults only `idx & 15`, so the index has no bits left to select a byte; the
  sibling probe `pq2lutsplit.c` settles it by construction.)
* Angle 5 (is the ceiling now FMA?) is answered in section 6: **no**, the FMA
  network is only 25% of V12's time; the ceiling is the per-block dependency
  chain.
* The whole V3/V7/V8/V9 family, and the IEEE-754 `OR 0x4B000000` bias idea, are
  strictly losing propositions: they all replace 1 op by 2-4 ops. (The bias
  route additionally requires a per-lane subtraction of `2^23 * x`, otherwise
  the accumulated constant destroys the result by catastrophic cancellation --
  measured twice over this project.)

---

## 3. What V12 actually changes

V12 = V10's regrouping (dot product by 2-bit field, `x` permuted once per
matvec call) **plus two instruction-count reductions in the LUT gather**.

### (a) Fold the block scale `d` into the LUT (8 multiplies -> 2)

`permutevar(code, LUT)` returns `LUT[code & 15]`. If the LUT vector itself is
pre-multiplied by the block scale `d`, the gather returns `(code-1)*d` -- the
weight -- so the eight `vmulps` that applied `d` afterwards disappear and are
replaced by **two** multiplies that scale the two LUTs once per block.
Numerically identical: the value is still the single product
`(float)(code-1) * d` of the same two exactly-representable operands.

### (b) Two LUTs instead of six shifts (6 `vpsrld` -> 2)

The index `permutevar` consults is `code & 15` -- the four low bits of the code
byte, i.e. **field 0 in bits 0-1 and field 1 in bits 2-3**. So one LUT
`LA[j] = (j&3)-1` extracts field 0, and a *second* LUT `LB[j] = ((j>>2)&3)-1`
extracts field 1 **from the same unshifted index**. Fields 2 and 3 need one
shift by 4 (which produces both), after which the same two LUTs serve them.
That is 2 shifts per block instead of 6, for one extra live register.

### per 128 weights

```
                    instructions emitted per 128 weights (icc -mmic -O3 -S)
  V1    16 vpermd + 8 vpsrlvd + 8 vmulps + 8 vfmadd + 4 vmovaps + 4 unpack
  V10c   8 vpermd + 6 vpsrld  + 8 vmulps + 8 vfmadd + 6 vmovaps + 4 unpack  = 40
  V12    8 vpermd + 2 vpsrld  + 2 vmulps + 8 vfmadd + 6 vmovaps + 4 unpack  = 30/31
```

`srccount.py` attributes each emitted instruction to its source line, so these
counts are from the actual loop bodies, not from the source. The design and the
disassembly agree: V12 has exactly the intended mix (8 gathers, 2 shifts, 2
scale multiplies), and the compiler emitted nothing extra beyond four
register-move `vmovaps` in the FMA block.

V12's per-block cost is ~107 cycles at 1238 MHz versus V10c's ~120 and V1's
~138, and the measured ratio matches the instruction ratio (30/40 = 0.75 ->
1.33x for the inline form, see section 7).

---

## 4. Cost attribution (`pq2v13`, one change per probe)

```
  V12  full candidate                   1229.5 M w/s   128.9 cyc/block
  Q1   no per-block fp16 decode          1691.4         93.7   (+37%)
  Q5   no LUT-scaling multiplies         1723.2         92.0   (+40%)
  Q10  block scale from a vector         1559.7        101.6   (+27%)
  Q2   8 adds instead of 8 permd         1228.6        129.0   (+0%)
  Q4   ONE LOAD16 instead of two         1273.2        124.5   (+3.6%)
  Q6   no shifts                         1412.4        112.2   (+15%)
  Q7   no xp loads                       1337.9        118.4   (+8.8%)
  Q8   8 permd + 8 FMA + 8 loads only    3127.6         50.7   (+154%)
  Q9   8 FMA + 8 xp loads only (floor)   4864.7         32.6   (+296%)
```

Two things stand out:

* **Q2: the permutes are free.** Removing them entirely from the accounting
  (Q8 vs Q9) prices all 8 at only 18 cycles.
* **Q1/Q5/Q10: the per-block scale handling is the single biggest cost.** The
  chain `load 2 bytes -> branchy scalar fp16 decode -> vbroadcastss -> 2 LUT
  multiplies -> 8 permutes -> 8 FMAs` is serialised on an in-order core, and
  ICC does not software-pipeline the block loop. Breaking that chain is worth
  1.38-1.40x -- more than any instruction-count change.

This is exactly why the earlier probes (`P_A`, `P_C` in `pq2v12.c`) were
misleading and were redone: `P_A` had silently dropped the LUT multiplies too
(it used the unscaled LUTs) and `P_C` had silently removed four permutes.

---

## 5. Latency-hiding attempts (what worked and what did not)

| variant | idea | result |
|---|---|---|
| V14b | 2 blocks in flight (same binary) | +17% over V12 inline |
| V14d | prescale the row's 40 block scales into an array | +20% over V12 inline |
| V14c | scale the `x` operand instead of the LUT (8 muls, permutes off the chain) | **0.96x -- a loss** |
| V14e/f | x-scale + prescale / + 2 blocks | +14% / +5% -- worse than V14d |
| V15 K=2..8 | K blocks in flight, same binary, same run | K=1 1466, K=2 1316, K=3 1362, K=4 1374, K=5 1389, K=8 1427 -- **all worse than K=1** |

Manual multi-block unrolling hurts: the extra live state costs more than the
extra ILP buys. Prescaling the row's scales is neutral-to-slightly-positive once
the code structure is right, and it carries a 16 KB stack array, so it is not
worth the integration risk.

---

## 6. Where the remaining time goes, and how much headroom is left

```
  per 128 weights, cycles at 1238 MHz
  32.6  8 FMAs + 8 xp loads (no decode at all)          <- absolute floor
  50.7  + the 8 LUT gathers                             (+18 for 8 permutes)
 106.0  + 2 LOAD16, 2 shifts, 2 scale muls, scalar decode, loop overhead  <- V12
```

So ~55 cycles per block are spent on the load/scale/gather pipeline rather than
on arithmetic. The arithmetic floor (32.6) is 3.2x the V1 baseline, but it
cannot be reached: a correct decoder must materialise 128 float weights per
block, which is 8 gathers, and it must obtain 32 code bytes per block as
16 dwords each.

The concrete remaining targets, in the order I would attack them:

1. **A batched, branchless fp16 scale decode.** Q10 shows that replacing the
   scalar branchy routine with a single permute out of a vector of decoded
   scales recovers 27 of the 35 cycles that Q1 recovers. KNC's
   `bonsai_fp16_to_fp32` is ~12 instructions with a data-dependent subnormal
   loop; a vectorised decode of 16 scales at a time (one 32-byte load + integer
   ops, or `vcvtph2ps` if it links) plus one permute per block should capture
   most of that. This is the largest single identified win left (~1.25x on top
   of V12 in the best case).
2. **Cheaper byte->dword widening.** Q4 prices a whole `LOAD16` at 4.4 cycles
   (2-3 instructions), so the 2 LOAD16 are only ~9 cycles of the 106. Not much
   left there; the earlier "LOAD16 costs 22 cycles" reading was a confounded
   probe.
3. **Software pipelining the block loop by hand** (compute block b+1's scale and
   LUTs while block b's gathers run). V15 shows that simply duplicating the body
   does not do this; it needs explicit reordering, and ICC may reorder it back.

A realistic target for a further round is 1.7-1.9x; >2x would require a way to
materialise more than 16 weights per gather, and the LUT arithmetic
(`idx & 15` selecting one dword) rules that out.

---

## 7. Shape dependence (measured, and it matters for integration)

The same binary at three shapes (all exact at every shape; the verification is
shape-independent):

```
  ratio vs V1             n=512 d=512    n=5120 d=512   n=17408 d=64
  V10c                       1.151          1.136-1.142     1.049-1.052
  V12 inline                 1.309          1.302-1.318     1.116-1.122
  V12 helper                 1.504          1.556-1.570     1.204-1.210
  V12 inline+pre             1.428          1.551-1.575     1.265-1.268
  V12 helper+pre             1.428          1.544-1.564     1.265-1.268
  V12 pf8 / pf16 / pf32      1.402          1.417-1.470     1.182-1.189
```

* The widest tensor (`n = 17408`) is the least favourable: 1.20-1.27x instead of
  1.55x. The likely cause is the `xp` array: at n=17408 it is 70 KB and no longer
  fits in KNC's 32 KB L1, so every `xp` load now comes from L2, while V1 reads a
  plain `x` it can at least prefetch sequentially. V1 itself also drops from 934
  to ~476 M weights/s there, so part of it is a property of the shape, not of
  V12.
* The obvious fix, which I did **not** implement (it is an integration decision,
  not a decoder decision): build and consume `xp` in L1-sized tiles
  (e.g. 4096 floats) and run all d rows over each tile before moving on, so the
  permuted input never leaves L1 and its build cost is amortised the same way.
* `helper+pre` is the most consistent choice across shapes (1.43 / 1.55 / 1.27);
  plain `helper` is better at the small shape (1.50) and worse at the widest
  (1.20). The differences between them are ~5%, i.e. of the same order as the
  run-to-run spread -- the factor that really matters is `helper` vs `inline`
  (19%).

---

## 8. Caveats the integrator must know

1. **The absolute ratio depends on code structure, not only on the algorithm.**
   The *same* V12 source measures **1.30x** when the block body is written out
   inline and **1.55-1.58x** when it is a small `static inline` helper (or when
   the row scales are prescaled, or when explicit `_mm_prefetch` is added at
   distance 8-32 blocks). The disassembly explains it: the faster forms contain
   compiler-inserted `vprefetch0/vprefetch1` that the plain inline form lacks;
   explicit prefetch alone recovers 1.42-1.47x, so prefetching is part of it and
   scheduling is the rest. **After integrating, disassemble the inner loop and
   check for the prefetch and the 8/2/2/8/6/4 mix; if the ratio comes out
   ~1.30x, the algorithm is still right and only the schedule is missing.**
2. **V12 needs an `xp` array** (a permuted copy of `x`, exactly `n` floats) built
   once per matvec call, exactly as V10 does. Cost measured earlier: 0.041-0.056
   ms per call at n=5120, i.e. 1.8% of the row work at d=512.
3. **Summation order changes relative to V1** (the dot product is regrouped by
   2-bit field). V12 is bit-identical to V10c; V12 vs V1 differs only by
   floating-point association (~1e-7 relative on real weights), which is why the
   end-to-end check must include `l_out-63`, as with V10.
4. `n` must be a multiple of 128 for the `xp` mapping (true for every tensor in
   this model).
5. **Measurement caveat:** the same kernel measured 1216 M w/s in `pq2v12.c` and
   1461 M w/s in `pq2v15.c`; the V1 reference moved from 934 to 718 in the same
   pair. That is why all final numbers come from `pq2v16.c`, which contains
   every kernel in one binary and prices them in both orders.

---

## 9. Files

```
src/bonsai/pq2prim.c, pq2prim2.c, pq2prim3.c   primitive cost tables / machine model
src/bonsai/pq2v12.c      V12 + first verification + (confounded) probes
src/bonsai/pq2v13.c      cost attribution, one change per probe
src/bonsai/pq2v14.c      latency hiding (verification harness bug, pricing valid)
src/bonsai/pq2v15.c      V12 + prescale + K-blocks-in-flight sweep
src/bonsai/pq2v16.c      controlled head-to-head, one binary, two pass orders  <- use this
src/bonsai/srccount.py   per-source-line instruction counts from icc -S
src/bonsai/pq2v12_card_evidence.txt            raw card output for everything above
```
