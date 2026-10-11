# PQ2_0 vectorised dequantiser for KNC (k1om) — final report

> ## ⚠️ 三处更正（本文件写于集成之前）
>
> **1. `BONSAI_IMCI_PQ2_0` 是幽灵宏。** 本文件多处提到用它开关内核 —— 那个宏**已不存在**。
> 现在的逻辑是：**k1om 上 IMCI 内核就是默认**（`#ifndef BONSAI_FORCE_SCALAR_PQ2_0`），
> 退路宏是 **`BONSAI_FORCE_SCALAR_PQ2_0`**。宿主机构建始终走标量参考。
> 详见 `src/bonsai/pq2_0.h` 的 `#ifndef BONSAI_FORCE_SCALAR_PQ2_0`。
>
> **2. 越界读量是 48 字节，不是 96。** 本文件（以及当时 `pq2_0.h` 的注释）写的是"约 96"，
> 那是保守估计。卡上对 `PROT_NONE` 守卫页实测：
> 足迹 `[blk, blk+81]`，即最多越过 34 字节块末尾 **48 字节**，且**全部 64 种源对齐同阈值**。
> 另外它最多还会向块**之前**读 61 字节。`gguf.c` 现在 `malloc(file_size + 64)` 并清零尾部。
>
> **3. 内核"默认禁用"的说法已过时。** 集成后内核是 k1om 默认启用；且它现在的
> 8-FMA 主体已修复了"每 block 只覆盖 64/128 个权重"的缺陷（那曾使结果恒为一半）。

Everything below was measured **on the card** (Xeon Phi 7120P, `mic0`,
192.168.1.100, MPSS 3.8.6, icc 2017.0.109 `-mmic -O2 -std=c11`).
No claim here comes from reading a header.

Artifacts:

| file | what |
|---|---|
| `src/bonsai/pq2_0.h` | `pq2_0_dequant_block_imci()` — the decoder (inside `#ifdef __MIC__`) |
| `src/bonsai/pq2imci.c` | correctness + throughput harness |
| `src/bonsai/pq2matvec.c` | full matvec harness (build with/without `-DBONSAI_IMCI_PQ2_0`) |
| `src/bonsai/intr_probe.c` | per-intrinsic link/semantics probe (one `-DTRY_<name>` per candidate) |
| `src/bonsai/pq2load2.c`, `pq2load3.c` | load-path alignment/boundary isolation (one test per process) |
| `tools/probe_links.ps1` | drives the probe: compile once per candidate, the **linker** decides |
| `tools/mic.ps1` | scp (`-O`, chacha20-poly1305) + `chmod +x` + run on the card |

---

## 1. What actually links (40 pass / 47 fail)

Built with `-D__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES`; without it many
failures show up as a misleading *compile* error (`a value of type "int" cannot
be used to initialize an entity of type "__m512i"`, i.e. implicit declaration)
instead of the real link verdict.

**LINKABLE (40)** — `_mm512_`:
`permutevar_epi32`, `mask_permutevar_epi32`, `swizzle_epi32`, `shuffle_epi32`,
`permute4f128_epi32`, `slli/srli/srai_epi32`, `sllv/srlv_epi32`,
`and/or/xor/andnot_si512`, `add/sub/mullo/abs/min/max_epi32`,
`mask_blend/mov/set1_epi32`, `cmpeq/cmplt_epi32_mask`,
`load_epi32`, `mask_load_epi32`, `store_epi32`,
`extload_epi32` (NONE and UINT8), `mask_extload_epi32`,
`extloadunpacklo/hi_epi32` (NONE and UINT8), `mask_extloadunpacklo_epi32`,
`i32extgather_epi32`, `alignr_epi32`,
`set1_epi8/16`, `set_epi32/64`, `setzero_si512`.

**DECLARED BUT NOT IN THE RUNTIME** (`undefined reference to ...` at link time) —
i.e. every textbook 2-bit-expansion primitive is dead on this card:
`_mm512_cvtepi32_ps` → **`_mm512_cvt_roundepi32_ps`** (the named blocker),
the whole `unpacklo/hi_epi8/16/32/64` family, `shuffle_epi8` (vpshufb),
`shuffle_i32x4`, `slli/srli_epi16`, `mullo_epi16`, `madd_epi16`,
`maddubs_epi16`, `movm_epi32`, `maskz_set1/mov_epi32`, `maskz_load_epi32`,
`permutexvar_epi32`, `permutex2var_epi32`, `permutevar_ps`,
`sub_epi8`, `bslli/bsrli_epi128`, and **every** `*loadu*`/`*storeu*` form
(there is no unaligned plain load on k1om).

**REJECTED BY THE COMPILER** on target:
`_mm512_cvtepu8_epi32` → `catastrophic error: Function contains unsupported
data types or intrinsics on target`; likewise `cvtepi8_epi32`,
`cvtepu16_epi32`, `cvtepu8_epi16`, `cvtepi16_epi32`.

Exact linker output for the two named blockers:
```
intr_probe.c:(.text+0x1b1): undefined reference to `_mm512_cvt_roundepi32_ps'
intr_probe.c:(.text+0x3d2): undefined reference to `_mm512_unpacklo_epi8'
intr_probe.c:(.text+0x233): undefined reference to `_mm512_slli_epi16'
intr_probe.c:(.text+0x1bc): undefined reference to `_mm512_mullo_epi16'
intr_probe.c:(.text+0x3aa): undefined reference to `_mm512_madd_epi16'
```

### Measured semantics (they are not the AVX-512 semantics)

* `_mm512_permutevar_epi32(idx, data)` — **operand order is (index, data)**
  (unlike AVX-512's `_mm512_permutexvar_epi32`) and it is a **full cross-lane
  16-entry dword gather**: `result[i] = data[idx[i] & 15]`. Verified with
  `data = {10,11,12,13,20,…}`, `idx = {0,1,2,3,0,…}` → `2 3 0 1 | 0 1 2 3 | …`.
* `_mm512_swizzle_epi32(v, BADC)` = `_mm512_shuffle_epi32(v, _MM_PERM_BADC)`
  = per-128-bit-lane `[d1,d0,d3,d2]`.
* `_mm512_permute4f128_epi32(v, BADC)` permutes **128-bit lanes**.
* `_mm512_alignr_epi32(a,b,1)` = `((b:a) >> 32).low512` = `[b1..b15, a0]`
  (full cross-lane dword shift).
* `_mm512_srlv_epi32` is a logical (unsigned) per-lane variable shift.

### Load alignment (found by fault isolation, `pq2load2.c` / `pq2load3.c`)

| form | requirement |
|---|---|
| `_mm512_load_epi32`, `_mm512_extload_epi32` NONE | 64-byte aligned — else **SIGSEGV** |
| `_mm512_extload_epi32` UINT8 | 16-byte aligned — else **SIGSEGV** |
| `_mm512_extloadunpacklo_epi32` NONE | 4-byte aligned (offsets 1,2,3 fault) |
| `_mm512_extloadunpacklo/hi_epi32` UINT8 | **any byte offset** |

The last one still has a trap: in UINT8 mode `lo(p)` delivers only the bytes
`[p .. end of the 64-byte aligned block containing p]`; **everything past that
boundary is taken from the `old` register**. With `old = 0` the tail is
silently ZERO — no fault. Measured at `p = base+52`:

```
lo(+52) ALONE : 109 116 123 130 137 144 | 179 186 0 0 0 0   <- 4 bytes silently lost
lo(+52)+hi(+64): 109 116 123 130 137 144 | 179 186 193 200 207 214   <- correct
```

This was the real cause of the first, silently-wrong version of this decoder:
a PQ2_0 payload is `qs = blk+2` with a 34-byte block stride, so `qs & 63 > 32`
for roughly half of all blocks, and ~10 of every block's 128 weights came back
zero. It is worth stressing that **this produced no crash and no error** — only
the element-by-element comparison caught it.

Because `qs` is only ever 2-byte aligned, the NONE (packed 64-byte) form of the
unpack load is unusable, so the decoder uses the UINT8 form and widens to one
byte per dword *in the load*.

---

## 2. The decoder

`src/bonsai/pq2_0.h`, `pq2_0_dequant_block_imci()`:

```c
#define PQ2_IMCI_LOAD16(p) \
    _mm512_extloadunpackhi_epi32( \
        _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), \
            (const void *)(p), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE), \
        (const void *)((const uint8_t *)(p) + 64), \
        _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)

static inline void pq2_0_dequant_block_imci(const uint8_t *blk, float *out) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    const __m512i  lut = PQ2_IMCI_LUT_BITS();   /* 16 dwords = float bits of (i&3)-1 */
    const __m512i  shf = PQ2_IMCI_SHIFT();      /* {0,2,4,6} x4 */
    const __m512  vd  = _mm512_set1_ps(d);
    const __m512i  c0  = PQ2_IMCI_LOAD16(qs);      /* dword i = qs[i]    */
    const __m512i  c1  = PQ2_IMCI_LOAD16(qs + 16); /* dword i = qs[16+i] */
    int m;
    for (m = 0; m < 4; ++m) {   /* weights   0.. 63 */
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
        const __m512i v = _mm512_srlv_epi32(b, shf);
        const __m512  g = _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut));
        _mm512_store_ps(out + 16 * m, _mm512_mul_ps(g, vd));
    }
    for (m = 0; m < 4; ++m) {   /* weights  64..127 */
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
        const __m512i v = _mm512_srlv_epi32(b, shf);
        const __m512  g = _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut));
        _mm512_store_ps(out + 64 + 16 * m, _mm512_mul_ps(g, vd));
    }
}
```

* `pq2_imci_idx(m)` = `{4m x4, 4m+1 x4, 4m+2 x4, 4m+3 x4}` broadcasts the four
  code bytes of group `m` to four lanes each.
* `shf = {0,2,4,6}` repeated extracts the 2-bit field of each lane; element
  `16m+e` is field `e%4` of code byte `4m+(e/4)`.
* The `int -> float` step — the original blocker — is **a shuffle, not a
  numerics problem**: `_mm512_permutevar_epi32(v, lut)` with a 16-entry table
  whose entry `i` is the IEEE-754 bit pattern of `(float)((i & 3) - 1)`. Because
  the table has 16 entries indexed by `(byte >> 2k) & 15`, the usual `& 3` is
  not even needed. No `cvtepi32_ps`, no bias arithmetic, one instruction.

Cost: **4 loads + 8 x (permutevar + srlv + permutevar + mul_ps) + 8 stores per
128 weights** = 4 vector ops + 1 store per 16 floats.

### Alternative that was measured and rejected

The IEEE-754 bias identity works on this card and is **exact** here — with
`k = code & 3 ∈ {0,1,2,3}` there is no negative operand, so the "`-1` came out
as `-0`" failure reported earlier does not reproduce for this use:
`(q-1) == bitcast_ps((code & 3) + 0x4B000000) - 8388609.0f`.
But it is **2.2x slower** (290 vs 1255 M weights/s) because it replaces one
`vpermd` with three 512-bit integer-ALU ops (`vpandd`, `vpaddd`) plus a
`vsubps`, and on KNC the 512-bit integer ALU and the variable shift are the
slow path. The LUT stays.

---

## 3. Correctness — element by element

`pq2imci.c`, `-mmisc`… build: `icc -mmic -O2 -std=c11 src/bonsai/pq2imci.c -o pq2imci -lm`

```
=== 0) sample block (d = 1.0) ===
  raw qs[0..3] = 0x03 0x0A 0x11 0x18
  scalar[0..7] = 2 -1 -1 -1 1 1 -1 -1
  vector[0..7] = 2 -1 -1 -1 1 1 -1 -1
  expect       = 2 -1 -1 -1 1 1 -1 -1

=== 1) _mm512_extloadunpacklo/hi_epi32 UINT8: 16 bytes at any offset ===
  offsets 0..63 x 16 lanes  : all correct (including every 64-byte boundary crossing)

=== 2) pq2_0_dequant_block_imci vs pq2_0_dequant_block ===
  4000 random blocks x 64 source alignments (block start 0..63)
  elements compared      : 32768000
  max |scalar - vector(LUT)|  : 0   mismatches 0
  max |scalar - vector(BIAS)| : 0   mismatches 0
  verdict                : BOTH EXACT (bit-identical)

=== 3) whole-matrix decode  n=10240 d=5120  (52428800 weights, 13.28 MB packed) ===
  weights compared       : 52428800
  max |scalar - vector|  : 0
  mismatching weights    : 0
  verdict                : EXACT over the entire matrix
```

* 32,768,000 comparisons across **all 64 source alignments** (`qs` covers every
  residue mod 64, odd and even, including every 64-byte boundary crossing):
  **max |diff| = 0**, 0 mismatches.
* The whole 10240x5120 matrix (52,428,800 weights), finite random fp16 scales
  (sub/normal, both signs), random 2-bit codes: **max |diff| = 0**,
  0 mismatches. Exactness is expected, not luck: `(q-1)` is one of
  `{-1,0,1,2}` and both paths compute the same single multiply.
* A third, deliberately un-optimisable `volatile` scalar reference
  (`deq_naive`) is cross-checked too, so a mismatch would be attributed
  correctly rather than assumed to be the vector path's fault.
* Earlier runs that *did* mismatch (2658/32768 and 4300993/52428800) are what
  exposed the `lo()` zero-fill trap described above.

## 4. Throughput — single thread

Same binary, same buffers, `pq2_0_dequant_block` vs `pq2_0_dequant_block_imci`.

| working set | scalar | vector (LUT) | speedup |
|---|---|---|---|
| 10240x5120 = 52.4M weights, 13.28 MB packed (streams from DRAM, L2 = 512 KB) | 3.957 s, 106.0 M w/s | 0.711 s, 589.7 M w/s | **5.56x** |
| 1024x32 = 32 K weights, 256 KB packed (fits L2 → compute bound) | 112.7 M w/s | 1254.4 M w/s | **11.13x** |

The gap between 11.1x and 5.6x is main-memory streaming: the IMCI path is fast
enough that the 13.28 MB working set no longer fits in cache. `vpermd` and
`vloadunpackld` are both microcoded on KNC, so 16 `vpermd` per 128 weights is
the remaining limit — not memory.

## 5. Wiring into the IMCI matvec (opt-in) and its measurement

`pq2_0_matvec_knc` (behind its **existing** `#ifdef BONSAI_IMCI_PQ2_0` guard)
now calls `pq2_0_dequant_block_imci` instead of `pq2_0_dequant_block`. The only
change inside the guarded branch is that one call; the 8-FMA body, the four
accumulators and the tail loop are untouched. **The default is untouched**:
`#define pq2_0_matvec_knc pq2_0_matvec_ref` still stands in the
non-`BONSAI_IMCI_PQ2_0` branch (line 365) and in the host branch (line 369).

`pq2matvec.c`, n=10240 d=5120, 3 reps, single thread (built without `-openmp`,
so the `#pragma omp parallel for` is inert):

```
########## default build (scalar kernel) ##########
  pq2_0_matvec_ref  :   0.4993 s    105.01 M weights/s
  pq2_0_matvec_knc  :   0.5044 s    103.95 M weights/s     speedup 0.99x

########## -DBONSAI_IMCI_PQ2_0 ##########
  pq2_0_matvec_ref  :   0.5017 s    104.51 M weights/s   (scalar decode + scalar MAC)
  pq2_0_matvec_knc  :   0.1008 s    520.06 M weights/s
  matvec speedup    : 4.98x
  |y|max = 3.36861e+06
  knc - ref : max abs diff = 1, max REL diff = 4.16e-04, rows with rel > 1e-3 = 0 / 5120
  IMCI matvec matches the scalar reference (float summation-order noise only)
```

So the full matvec goes **0.502 s → 0.101 s, 4.98x**, and the IMCI kernel now
agrees with the scalar reference (the residual 4e-4 relative difference is
float summation order: 4 partial accumulators vs one sequential sum — the
same class of difference the earlier fix reported). The IMCI path is still
opt-in, as required.

## 6. Constraints respected

* Host builds still work: `zig cc -O2 -std=c11 -I src/bonsai -c
  src/bonsai/model.c` and `... gguf.c` both exit 0
  (`D:\zig\zig-x86_64-windows-0.17.0\zig.exe`). Everything added to `pq2_0.h`
  is inside `#ifdef __MIC__`.
* `pq2_0_matvec_ref` is byte-for-byte unchanged, `#pragma omp parallel for
  schedule(static)` included.
* The IMCI matvec is still disabled by default.

## 7. Caveats for the next reader

* The decoder reads the two 64-byte aligned blocks that bracket each payload,
  so it touches up to ~96 bytes past the end of a block. Callers must leave
  readable slack after the weights buffer (the 6.7 GB model does; the harnesses
  here allocate 512 extra bytes). If a caller ever maps weights at the very end
  of a mapping, handle the last block of the last row specially.
* `_mm512_extloadunpacklo/hi_epi32` with `_MM_UPCONV_EPI32_NONE` needs 4-byte
  alignment and the UINT8 form needs none — but the UINT8 form is asymmetric
  (`lo` zero-fills past the aligned block, `hi(p+64)` completes it). Do not use
  `lo` alone.
* Reproduce with `tools/probe_links.ps1` (link verdicts) and `tools/mic.ps1`
  (upload + run). Both need PowerShell execution policy bypass for the process:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
