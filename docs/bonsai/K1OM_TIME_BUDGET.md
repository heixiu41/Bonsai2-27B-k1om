# Where the wall-clock time actually goes — measured time budget for Bonsai 2 27B on the 7120P

Instrument: an instrumented copy of the engine in `profile/` (`prof.h`, `prof.c`,
`model.c`, `pq2_0.h`, `main_k1om.c`), built with `-DBONSAI_PROF=1`.
The verified originals in `src/bonsai/` are **byte-for-byte untouched**
(`pq2_0.h` md5 `20C6F16286745907469AC1757A72FB99`, `model.c` md5
`4FE838CC9109D7454A57692E4C162372`, `gguf.c` md5 `F82CC4498C6A2E9C412D42E30089138C`,
`main_k1om.c` md5 `96C836C7BAD2DBA1F25A717B155F0742`; zero probe macros in any of them).

All numbers below are **measured on the card** (mic0, 192.168.1.100), 122 threads
unless stated. The engine is numerically unchanged: every run reported
`top-1 = 11751`, `rank 0/248320`.

---

## 0. The answer in one line

**It is (B), overwhelmingly — and the thing it is *not* is the decoder.**

The PQ2_0 decoder/matvec is **11.6 %** of the per-token time. **74.5 %** is the
serial GDN glue, of which a single scalar triple loop — the DeltaNet recurrence —
is **70.2 %**. The thread-count sweep proves it: the recurrence takes
**11.85–13.76 s per 5-token prefill at 1, 30, 61, 122 and 244 threads** — it is
completely flat, i.e. purely serial.

Parallelising that one loop's 48 independent heads (one `#pragma`, numerics
unchanged) took the engine from **0.291 tok/s to 0.954 tok/s — 3.28×**.

---

## 1. Measured time budget

Full forward run (5-token prefill + 4 decode tokens = 9 tokens), 122 threads.
`tot` is the sum over threads; for a serial region that *is* the wall time, and
for an OpenMP region only the master's instantaneous interval is counted.

### Per token, and over the 9-token run

| region | 9 tokens (s) | % of span | 1 token (s) | decode-only (s/tok) |
|---|---|---|---|---|
| **whole span** (reset+prefill+scan+decode) | **31.384** | 100 % | 3.487 | **3.433** |
| `bonsai_forward` (all 9) | 31.076 | 99.0 % | 3.453 | 3.400 |
| transformer blocks (64×) | 30.932 | 98.6 % | 3.437 | — |
| **GDN blocks (48×)** | **25.897** | **82.5 %** | **2.877** | **2.858** |
| full-attention blocks (16×) | 0.666 | 2.1 % | 0.074 | 0.119 |
| FFN blocks (64×) | 4.366 | 13.9 % | 0.485 | — |
| final norm + output head | 0.138 | 0.4 % | 0.015 | — |
| embed_row | 0.007 | 0.02 % | 0.001 | — |

### Inside the GDN block (48 per token) — the hot 82.5 %

| region | 9 tokens (s) | % of span | per token (s) |
|---|---|---|---|
| **GDN block total** | 25.896 | 82.5 % | 2.877 |
| **DeltaNet recurrence (serial)** | **22.031** | **70.2 %** | **2.448** |
| W_qkv @ had(xb) `[10240×5120]` | 1.194 | 3.8 % | 0.133 |
| causal depthwise conv (serial) | 0.993 | 3.2 % | 0.110 |
| W_beta/W_alpha @ xb (+4 small dequants) | 0.765 | 2.4 % | 0.085 |
| ├ the two scalar `matvec_f32` calls | 0.379 | 1.2 % | 0.042 |
| └ beta/alpha activation | 0.009 | 0.03 % | 0.001 |
| per-head gated RMSNorm + silu(z)*o (serial) | 0.339 | 1.1 % | 0.038 |
| W_ssm_out @ had(.) `[5120×6144]` | 0.520 | 1.7 % | 0.058 |
| attn_norm RMSNorm + L2 norm + residual | 0.051 | 0.2 % | 0.006 |

(The GDN children sum to 25.88 s against a 25.896 s parent: **reconciles** —
the sub-region timings are trustworthy.)

### Inside the full-attention block (16 per token)

| region | 9 tokens (s) | per token (s) |
|---|---|---|
| block total | 0.666 | 0.074 |
| W_q @ had(xb) `[12288×5120]` | 0.203 | 0.023 |
| W_attn_output @ had(.) `[5120×6144]` | 0.168 | 0.019 |
| W_k / W_v @ had(xb) (2× `[1024×5120]`) | 0.212 | 0.024 |
| KV append + attention + softmax | 0.022 | 0.002 |
| per-head QK RMSNorm (256) | 0.021 | 0.002 |
| Q/gate deinterleave, RoPE, gate, residual | 0.038 | 0.004 |

### Inside the FFN (64 per token)

| region | 9 tokens (s) | per token (s) |
|---|---|---|
| FFN total | 4.366 | 0.485 |
| W_ffn_down @ had(.) `[5120×17408]` | 2.273 | 0.253 |
| W_ffn_gate @ had(xb) `[17408×5120]` | 0.981 | 0.109 |
| W_ffn_up @ had(xb) `[17408×5120]` | 0.994 | 0.110 |
| silu(gate)*up (serial) | 0.080 | 0.009 |
| post_attention_norm RMSNorm + residual | 0.034 | 0.004 |

### Inside `mm_folded` — the only place the PQ2_0 decoder runs

`mm_folded` is called **3609 times per 9 tokens (401/token)** and decodes
**230,385,254,400 weights** (25.60 G weights/token — matches the 26.87 G
nominal). *Every one of those calls is a Hadamard call* (all folded weights are
stored in the rotated basis), which is why `mm_folded` and the Hadamard total
both hit 3609 calls.

| region | 9 tokens (s) | % of span | per token (s) |
|---|---|---|---|
| **`mm_folded` grand total** | **6.644** | **21.2 %** | **0.738** |
| **└ PQ2_0 matvec kernel (the decoder)** | **3.651** | **11.6 %** | **0.406** |
| &nbsp;&nbsp;&nbsp;└ same, timed around the OpenMP region | 3.640 | 11.6 % | 0.404 |
| **└ Hadamard transform inside `mm_folded` (serial, no OpenMP)** | **2.968** | **9.5 %** | **0.330** |
| &nbsp;&nbsp;&nbsp;└ of which sign-multiply + copy | 0.099 | 0.3 % | 0.011 |
| (remainder: argument checks, 20 ms) | ~0.02 | 0.06 % | — |

Two independent probes (the master's `mm_folded` interval and the kernel's own
around-the-parallel-region interval) agree to **0.3 %** (3.651 vs 3.640 s),
which is the strongest single piece of evidence in this report.

### Grand total, per token, over all 64 layers

```
GDN blocks total               2.877 s   82.5%
  DeltaNet recurrence          2.448 s   70.2%   <-- SERIAL, THE BOTTLENECK
  W_qkv matvec                 0.133 s    3.8%
  conv                         0.110 s    3.2%
  beta/alpha + their dequants   0.085 s    2.4%
  W_ssm_out matvec              0.058 s    1.7%
  gated RMSNorm + silu          0.038 s    1.1%
  misc                          0.006 s    0.2%
FFN blocks total               0.485 s   13.9%
  W_ffn_down matvec             0.253 s    7.3%
  W_ffn_gate/up matvec          0.219 s    6.3%
  silu(gate)*up + norm + add    0.013 s    0.4%
full-attention blocks total    0.074 s    2.1%   (decode: 0.119 s)
  W_q/W_k/W_v/W_out matvec      0.065 s    1.9%
  attention + softmax           0.002 s    0.05%
output head + final norm       0.015 s    0.4%
embed_row                      0.001 s    0.02%
─────────────────────────────────────────────
TOTAL                          3.453 s   (measured 3.433 s in steady decode)
```

---

## 2. Thread-count scaling — this is what decides (A) vs (B)

Same binary, same run shape (5-token prefill), only `OMP_NUM_THREADS` varied.

| threads | prefill 5 tok (s) | tok/s | **DeltaNet recurrence (s)** | matvec kernel (s) | Hadamard (s) | conv (s) | W_qkv (s) | FFN blocks (s) | attn blocks (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1   | 185.19 | 0.027 | **11.891** | 170.396 | 1.577 | 0.526 | 22.357 | 124.886 | 9.446 |
| 30  | 21.54  | 0.232 | **11.883** | 6.772 | 1.576 | 0.527 | 1.239 | 5.850 | 0.614 |
| 61  | 18.24  | 0.274 | **11.846** | 3.525 | 1.571 | 0.523 | 0.843 | 3.470 | 0.439 |
| 122 | 17.28  | 0.289 | **12.230** | 2.061 | 1.647 | 0.549 | 0.709 | 2.416 | 0.366 |
| 244 | 18.84  | 0.265 | **13.564** | 1.741 | 1.997 | 0.672 | 0.947 | 2.227 | 0.379 |

Read the recurrence column: **flat to within run-to-run noise from 1 to 244
threads.** It is 45 % of the 1-thread run and 72 % of the 122-thread run. That
flattening *is* the answer to (A) vs (B): a large serial fraction exists, and
adding threads cannot touch it.

The prefill curve also flattens exactly as that predicts. Fitting a serial
fraction from the 1→122 scaling of the *total*: 185.19 → 17.28 s is a 10.7×
speedup, not 122×, which is a serial fraction of ~25 % *of the two biggest
serial items alone*.

The matvec, by contrast, does scale: 170.4 s → 1.74 s (98× on 122 threads), and
its members clearly show hyperthreading diminishing returns (61 threads capture
48 % of the 122-thread speedup; 244 threads buy only 1.18× over 122 and cost the
Hadamard 21 % more because 244 threads disturb the serial master).

---

## 3. The dominant region: why it is slow

### 3.1 What it is

`model.c:804-821`, executed 48×/token (once per GDN block):

```c
for (int h = 0; h < vh; ++h) {              /* vh = 48 value heads */
    ...
    for (int a = 0; a < st; ++a)            /* st = 128 */
        for (int i = 0; i < st; ++i)
            Sh[a*st + i] *= decay;          /* NO omp pragma */
    for (int i = 0; i < st; ++i) {
        float sk = 0.0f;
        for (int a = 0; a < st; ++a)
            sk += Sh[a*st + i] * khv[a];    /* stride-128 reads */
        const float d = bh * (vhv[i] - sk);
        float o = 0.0f;
        for (int a = 0; a < st; ++a) {
            Sh[a*st + i] += khv[a] * d;     /* stride-128 read+write */
            o += Sh[a*st + i] * qhv[a];
        }
        oh[i] = o * qscale;
    }
}
```

There is **no `#pragma omp` anywhere in `block_gdn`'s model glue**. The only
OpenMP in the entire token is inside `pq2_0_matvec_knc`.

### 3.2 Evidence that it is serial and unvectorised

1. **Flat thread scaling** (§2): 11.891 s at 1 thread, 11.846 s at 61, 12.230 s
   at 122, 13.564 s at 244. A parallel loop cannot do that.

2. **Arithmetic gives 244 ps per FLOP.** per token: 48 blocks × 48 heads ×
   128 × 128 × 3 multiply-adds ≈ 113 M multiply-adds; 22.031 s / 9 tokens =
   2.448 s per token → **21.6 ps per multiply-add**, i.e. ~15 cycles at
   1.238 GHz for one FMA that a vector unit would do 16-wide. The code is
   running ~240× slower than the core's vector peak.

3. **The compiler emits gathers, not clean vectors.** `icc -mmic -O2` on
   `model.c` (5570-line `block_gdn`) contains **64 `vgatherdps` instructions in
   retry loops**, annotated `Latency 45`:

   ```
   ..B9.45:
           vpcmpltd  %zmm2, %zmm5, %k3
           ...
   ..L442:
           vgatherdps (%r9,%zmm1,4), %zmm7{%k1}
           jkzd      ..L441, %k1
           vgatherdps (%r9,%zmm1,4), %zmm7{%k1}
   ..L441:
           vmulps    %zmm7, %zmm3, %zmm8
   ```

   `vgatherdps` is microcoded on KNC and is the worst instruction choice on this
   core. The recurrence's `Sh[a*st+i]` walks 128 rows 512 bytes apart, which is
   exactly the pattern ICC turns into a gather.

4. **`-O3` does not help.** Same probes, same binary pair, `-O2` vs `-O3`:
   recurrence 13.764 s → 13.255 s (−3.7 %), total 19.23 s → 18.66 s (−3 %),
   top-1 still 11751. So this is not a missing-optimisation-flag problem; the
   loop structure defeats the vectoriser.

5. **It is not bandwidth.** The recurrence's whole working set per block is
   48 heads × 128 × 128 × 4 B = 3.1 MB of state plus 2 × 24 KB of k/q — L2/L3
   resident. Measured state traffic is ~3.1 MB read + 3.1 MB write per block,
   i.e. ~300 MB/token over 2.45 s ≈ **0.12 GB/s** against a measured 196 GB/s
   available. The loop is 1600× off the bandwidth limit.

### 3.3 The clincher: parallelising the heads, measured

The 48 value heads are **provably independent**: head `h` touches only
`Sh = S + h*st*st` (its own state), `khv/qhv = k/q + (h%kh)*st` (read-only),
`vhv = v + h*st` (read-only), `beta[h]`, `g[h]`, `oh = o + h*st`. Nothing is
shared for writing.

I built that variant (`profile/model_headpar.c`) — **one added line**:

```c
const float qscale = 1.0f / sqrtf((float)st);
float *S = m->st.ssm_state + (size_t)il * vh * st * st;
#pragma omp parallel for schedule(static)      /* <-- the entire change */
for (int h = 0; h < vh; ++h) {
```

A/B, **same sources, same build flags, same run, serial vs parallel**:

| metric (9 tokens, 122 threads) | shipped (serial recurrence) | head-parallel |
|---|---|---|
| prefill 5 tokens | 17.33 s (0.289 tok/s) | **5.40 s (0.926 tok/s)** |
| **decode** | 13.73 s, **3.433 s/token, 0.291 tok/s** | **4.19 s, 1.049 s/token, 0.954 tok/s** |
| **DeltaNet recurrence** | **22.013 s** | **0.578 s** |
| GDN blocks total | 25.855 s | 4.395 s |
| FFN blocks | 4.385 s | 4.385 s |
| matvec kernel | 3.657 s | 3.633 s |
| Hadamard | 2.965 s | 2.963 s |
| top-1 / rank | 11751 / 0 | **11751 / 0 (identical)** |

**The recurrence: 22.013 s → 0.578 s = 38.1× faster. End-to-end: 3.28×.**
Every other region is unchanged to within 1 %, and the deterministic result
fields are byte-identical.

That is a 38× return on a one-line change, and it also *closes the case*: the
flattening curve was the recurrence, and the (B) hypothesis is confirmed by
direct ablation.

### 3.4 Second dominant serial item: the Hadamard (9.5 %)

`pq2_0.h::hadamard_forward` → `fwht_inplace` is a **serial, scalar** Walsh-
Hadamard transform over 1024-element blocks, called once per folded weight per
token (401×/token), performing 261,457,920 float add/sub per 9 tokens.

- Measured: **2.968 s / 9 tokens = 0.330 s/token (9.5 %)**.
- Thread-flat: 1.577 s at 1 thread, 1.997 s at 244 (it gets *worse* with more
  threads, because it is serial work on the master while 244 workers contend).
- Throughput: 29.05 M add/sub per token / 0.330 s = **88 M add/sub/s**, i.e.
  ~14 cycles per add/sub — vs a 1-cycle scalar add and a 16-wide vector unit.
  No OpenMP, no vectorisation (`for (h=1; h<n; h<<=1) for (i=0;...)` with a
  fixed in-block stride; nothing here hints ICC to vectorise the butterfly).
- It is **10× more expensive than the entire full-attention stack** (0.074 s/tok).

Also serial in the GDN glue: the causal conv (0.110 s/tok) and the per-head
gated RMSNorm + `silu(z)*o` (0.038 s/tok). The FFN's `silu(gate)*up` is 0.009
s/tok.

---

## 4. What fraction the PQ2_0 decoder accounts for — and whether more decoder work is worth anything

**The answer is 11.6–11.8 %. Stop optimising the decoder.**

Evidence, from the same runs:

- `PQ2_0 matvec kernel` = **3.651 s of 31.384 s = 11.63 %** of the 9-token span,
  and **3.657 s of 30.913 s = 11.83 %** of the transformer-block time.
- Cross-checked by two independent probes agreeing within 0.3 %
  (3.651 s around the call vs 3.640 s around the OpenMP region).
- Weight accounting closes exactly: 230,385,254,400 weights decoded = 25.60 G
  weights/token × 9 tokens, and the call count (3609) equals 401/token
  (the 401 folded tensors, verified from the report's call counter).

What each hypothetical decoder win is actually worth, at 122 threads:

| scenario | matvec time | total/token | gain | tok/s |
|---|---|---|---|---|
| **today** | 0.406 s | 3.433 s | — | 0.291 |
| V10c integrated (−11.8 % on the kernel) | 0.358 s | 3.385 s | **1.4 %** | 0.295 |
| decoder 2× faster | 0.203 s | 3.230 s | **5.9 %** | 0.310 |
| **decoder infinitely fast** | **0** | **3.027 s** | **11.8 %** | **0.330** |

Even a *free* decoder yields 0.330 tok/s. The entire V1→V10c line of work,
which occupied three agents and six optimisation rounds, is worth **1.4 %**
end-to-end. For comparison, the one-line `#pragma` above is worth **228 %**.

**Also note what the decoder is not:** it is not bandwidth-bound. It streams
61.2 GB of weight bytes per token (230.4 G weights × 2.125 bits) in 3.651 s =
**16.8 GB/s**, against 196 GB/s available — 8.6 % of bandwidth. It is not
bound by the memory round-trip (the docs already showed V1→V2 register-residency
buys nothing). It is issue/latency-limited at ~514 M weights/s/thread vs 918 M
single-threaded, which is why 244 threads buy so little.

**The `vpermd`-density conclusion in `PQ2_DECODER_COST_ANALYSIS.md` remains
correct as a statement about the kernel.** It is simply irrelevant to the
engine's speed, because the kernel is 12 % of the time.

---

## 5. What to fix first, in order, and what each is worth

Estimates below are measured for #1 and arithmetic projections for #2–#4.

| # | change | where | est. time after | est. tok/s | gain | risk |
|---|---|---|---|---|---|---|
| **1** | `#pragma omp parallel for schedule(static)` on the 48 independent value heads | recurrence loop, `block_gdn`, ~`model.c:781` | 1.05 s/token | **0.95** | **3.28×** | measured; numerics identical |
| **2** | parallelise `fwht_inplace`'s blocks across the 1024-element blocks of one call (each block is independent; the butterfly inside a block is not) — or vectorise the butterfly | `pq2_0.h`, `hadamard_forward`/`fwht_inplace` | ~0.75 s/token | **~1.3** | +40 % | low: same arithmetic, different order *between* independent blocks only |
| **3** | parallelise the conv channel loop + the gated RMSNorm/`silu` loop; vectorise `silu(gate)*up` | `block_gdn`, `block_ffn` | ~0.60 s/token | **~1.65** | +27 % | low |
| **4** | if these land, re-tune threads: with the serial work gone, 244 may finally beat 122 — and re-examine the matvec's per-thread 56 % efficiency | — | — | — | — | — |
| — | further PQ2_0 decoder work (V10c, bias trick, permute elimination) | `pq2_0.h` | 3.43 → 3.39 s | 0.295 | **1.4 %** | **not worth doing** |

Ordering rationale: #1 is done and verified. #2 is the next real serial block
and is >5× larger than the whole attention stack. Only after #1–#3 does the
decoder become a meaningful fraction, and even then #4's re-tune is worth more
than any decoder change.

**The realistic ceiling.** The irreducibly serial work at 122 threads is
recurrence 2.448 + Hadamard 0.330 + conv 0.110 + per-head RMSNorm/silu 0.038
+ attn QK-norm/RoPE/gate 0.004 + FFN `silu(gate)*up` 0.009 ≈ **2.94 s/token**,
which is within 14 % of the measured 3.433 s/token — that gap *is* the parallel
work (the matvec, the norms, the residual adds), and it is why the curve is
flat. With #1–#3 removed, what is left is the matvec (0.406 s/token at 122
threads; 0.152 s/token in the 244-thread sweep) plus the now-parallel remainder,
so **the engine should reach several tok/s without any decoder change at all**.
The head-parallel run already demonstrates the direction: 0.954 tok/s with
**only** #1 done, i.e. with the Hadamard (0.33 s/tok), conv (0.11 s/tok) and
FFN/norm elementwise work all still serial.

The caveat that matters for planning: these are *empty-machine* numbers. The
Hadamard at 244 threads measured 1.577 s (at 1 thread) → 1.997 s (at 244), i.e.
it degrades as workers contend with the serial master, so I would not promise
the same tok/s on a card that is also serving SCIF or a host process.

---

## 6. Negative results, uncertainty, and what I could not attribute

### Negative results (all measured, do not re-investigate)

- **The PQ2_0 decoder is not the bottleneck** (11.6–11.8 %). A free decoder
  gives +11.8 %; the V10c decoder gives +1.4 %.
- **The decoder is not bandwidth-bound**: 16.8 GB/s of 196 GB/s available.
- **The matvec does not scale well per thread**: 514 M weights/s/thread at 122
  threads vs 918 M single-threaded (56 %). 61→122 threads gives only 1.71× on
  the matvec; 122→244 gives only 1.18×. Hyperthreading contention, not
  bandwidth.
- **`-O3` is worth ~3 %** (recurrence 13.764→13.255 s, total 19.23→18.66 s) and
  changes no result. Not a substitute for the structural fix.
- **The Hadamard gets *worse* with more threads** (1.577 s at 1 → 1.997 s at
  244): extra workers contend with the serial master.
- **Full attention is nearly free** (0.074 s/token, 2.1 %), and everything
  inside it including softmax and RoPE is ≤0.023 s/token. RoPE is 0.005 s over
  9 tokens (0.016 %). **RoPE and softmax are not worth touching.**

### Uncertainty

- **Run-to-run spread on serial regions is ~±8 %.** The recurrence measured
  11.85–13.76 s across nominally identical 122-thread runs (11.883 at 30
  threads, 12.230 at 122, 13.764 in the `-O2` leg of the O3 test). The *trend*
  (flat) is far outside that spread; individual values should be read to two
  significant figures.
- **Instrumentation cost is small but not zero.** The same instrumented sources
  built without `-DBONSAI_PROF` (`prof_np`) ran 3.4103 s/token vs 3.4325 s/token
  with probes on at 122 threads — **~0.6 %**. Probe cost is ~2 clock reads per
  region entry; at ~25,000 entries/token and ~50 ns per pair that is ~1 ms/token
  plus lost inlining in the matvec.
- **120 vs 122 threads.** The reference run used 122 (2 threads/core on 61
  cores); a couple of runs above were taken at 120. The difference is inside the
  noise band and does not affect any conclusion.
- **Thread-local timing is only reported faithfully for master-executed
  regions.** `tot` for an OpenMP region is not meaningful and is not used for
  any conclusion; only the master interval (`PR_MM_MATVEC`, `PR_MM_MATVEC_REGION`)
  is. I say which is which wherever a number is quoted.
- **No `perf`, no hardware counters.** I did not check (`perf` is very likely
  absent on this BusyBox/2.6.38 target) and did not use PMCs. All claims about
  *why* (gathers, stride, issue limits) are supported by disassembly and by
  arithmetic against the measured times, not by counters. The one causal claim
  I consider fully proven is #1, because it was verified by A/B ablation, not by
  argument.

### What I could not attribute

- **The ~0.9 s/token (26 %) that is not the recurrence, the Hadamard, the conv
  or the matvec.** The instrumented regions reconcile for the big items
  (GDN children sum to the GDN parent; block children sum to the block parent),
  but the per-layer elementwise regions that do not have `PR_` sub-regions —
  the residual adds, the `f_silu`, the `memcpy`s, the L2/v copy, the logit scan,
  and the function-call overhead of 9 `snprintf`s per layer (576/token) — are
  inside `PR_GDN`'s or `PR_FFN`'s but not individually timed. The single largest
  unsplit item is `PR_GDN_BETA`'s 0.765 s/token minus its two measured children
  (0.379 + 0.009) ≈ **0.377 s/token (11 %)**, which is 4 `dequant_to_f32` calls
  on ~2.6 M elements per layer (22,528 weights/layer × 48 layers). Small
  dequantisation done 48× per layer per token is a plausible next target but I
  did not split it further.
- **Whether the recurrence can be made *vector*-efficient**, as opposed to
  merely parallel. Head-parallelism removed 97 % of its cost; the residual
  0.578 s / 9 tokens = 0.064 s/token is 1.9 % of the total, so the strided
  layout is no longer worth restructuring — unless #2–#4 land and it becomes
  visible again.

### Instrumentation pitfalls hit and fixed (so the next person does not repeat them)

These each produced a *plausible-looking but wrong* budget; all three are
documented in `profile/prof.h`:

1. **`__attribute__((cleanup))` on ICC/k1om ran the handler immediately**, so
   every protected region measured only the lifetime of its own scope object
   (16710 handler firings, ~0.5 µs per call, all regions ~0). Replaced with
   explicit start/stop calls.
2. **A file-scope `double pr_t0` anchor** gave every translation unit one shared
   timer, so nested `PR_BEGIN`s clobbered their parents: `PR_GDN` summed to
   14.5 s while the `PR_BLOCKS` probe enclosing it reported 0.061 s. Anchors are
   now per region id.
3. **`omp_get_wtime()` was unusable here**; the probe clock is
   `clock_gettime(CLOCK_MONOTONIC)` (which also needs `_POSIX_C_SOURCE`
   defined inside `prof.h`, before any system header, since `prof.h` is included
   first).

A fourth, unrelated trap: `profile/main_k1om.c` was destroyed once by a
PowerShell `Get-Content`/`Set-Content` round-trip (it replaced `i` with `n`
throughout). Use the `write`/`edit` tools, never PowerShell text pipelines.

---

## 7. Reproducing

```powershell
# build (instrumented)  -- sources in profile/, originals untouched
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
Push-Location profile
& $icc -mmic -O2 -std=c11 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -DBONSAI_PROF=1 `
       main_k1om.c model.c gguf.c prof.c -o D:\zigtest\k1om\prof -lm
Pop-Location
```

```sh
# on the card
export LD_LIBRARY_PATH=/tmp
/tmp/prof /tmp/bonsai.gguf forward 122      # full budget (5 prefill + 4 decode)
/tmp/prof /tmp/bonsai.gguf prefill 244      # one point of the thread sweep
BONSAI_PROF=0 /tmp/prof ...                 # probes disabled at run time
```

Files: `profile/prof.h` (probe API + region ids), `profile/prof.c` (accumulator,
report, per-thread entry dump), `profile/model.c` and `profile/pq2_0.h`
(instrumented engine, numerics unchanged), `profile/main_k1om.c` (entry point
with `prefill`/`decode`/`forward` modes), `profile/model_headpar.c` (the
verified head-parallel recurrence experiment), `profile/sweep.sh`, `profile/hp.sh`.

**A caveat on one report field.** The `tot/nthr` column in `prof_report` divides
by the thread count, which is meaningless for serial (master-only) regions.
Read the `tot` column for serial regions and the `max`/`entries` pair for
parallel ones.
