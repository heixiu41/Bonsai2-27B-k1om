# llama2.knc.c — Native Deployment Debugging Log for Xeon Phi 7120P (Knights Corner)

> Date: 2026-07-31 | Goal: run llama2 inference natively on the Xeon Phi 7120P (k1om)

## Table of Contents

- [1. Project Background](#1-project-background)
  - [Hardware Environment](#hardware-environment)
  - [Software Environment](#software-environment)
  - [Key Architectural Differences](#key-architectural-differences)
- [2. Code Change Summary](#2-code-change-summary)
  - [2.1 Protecting Critical Code Paths](#21-protecting-critical-code-paths)
  - [2.2 Build/Link Scripts](#22-buildlink-scripts)
- [3. Debugging Journey](#3-debugging-journey)
  - [3.1 Initial Approach: MIC Offload (abandoned)](#31-initial-approach-mic-offload-abandoned)
  - [3.2 Setting Up the Cross-Compile Environment](#32-setting-up-the-cross-compile-environment)
  - [3.3 Fixing Compile Errors](#33-fixing-compile-errors)
  - [3.4 Debugging Runtime Crashes](#34-debugging-runtime-crashes)
- [4. Root Cause Analysis: Weight Memory Alignment](#4-root-cause-analysis-weight-memory-alignment)
  - [4.1 Hypothesis](#41-hypothesis)
  - [4.2 Chain of Evidence](#42-chain-of-evidence)
  - [4.3 Affected Locations](#43-affected-locations)
- [5. Solutions](#5-solutions)
  - [Option A: Copy Weights to an Aligned Buffer in read_checkpoint (Recommended)](#option-a-copy-weights-to-an-aligned-buffer-in-read_checkpoint-recommended)
  - [Option B: Use Unaligned Load Intrinsics](#option-b-use-unaligned-load-intrinsics)
  - [Option C: Modify the Checkpoint File Format](#option-c-modify-the-checkpoint-file-format)
  - [Option D: Manually Skip Alignment Bytes After the Config Struct](#option-d-manually-skip-alignment-bytes-after-the-config-struct)
- [6. Current Status and Benchmarks](#6-current-status-and-benchmarks)
  - [6.1 Current Status](#61-current-status)
  - [6.1.1 Final Build Approach: Calling icl.exe Directly from PowerShell](#611-final-build-approach-calling-iclexe-directly-from-powershell)
  - [6.2 Benchmark: stories110M.bin (12 layers, dim=768, hidden_dim=2048)](#62-benchmark-stories110mbin-12-layers-dim768-hidden_dim2048)
  - [6.3 Benchmark: stories15M.bin (6 layers, dim=288, hidden_dim=768)](#63-benchmark-stories15mbin-6-layers-dim288-hidden_dim768)
  - [6.4 Scalar Version Sample Output](#64-scalar-version-sample-output)
  - [Completed Fixes](#completed-fixes)
- [7. Technical Notes](#7-technical-notes)
  - [7.1 Key KNC MVEX Intrinsic Mapping](#71-key-knc-mvex-intrinsic-mapping)
  - [7.2 PSXE 2017 /Qmic Compiler Behavior](#72-psxe-2017-qmic-compiler-behavior)
  - [7.3 WSL Interop Notes](#73-wsl-interop-notes)
  - [7.4 Known _mm512 Usage Sites (run.c MIC Path)](#74-known-_mm512-usage-sites-runc-mic-path)
- [8. File Inventory](#8-file-inventory)
- [9. Quick Command Reference](#9-quick-command-reference)
- [10. Matmul Precision Modes](#10-matmul-precision-modes)
  - [10.1 BFP16 Test Results (Unusable)](#101-bfp16-test-results-unusable)
  - [10.2 Int8 Quantization (Q8_0, runq.c)](#102-int8-quantization-q80-runqc)
  - [10.3 llama2-7b Deployment Plan](#103-llama2-7b-deployment-plan)
- [11. llama2-7b Download and Export](#11-llama2-7b-download-and-export)
  - [11.1 Prerequisites](#111-prerequisites)
  - [11.2 Export Commands](#112-export-commands)
  - [11.3 Export Format (v2 Q8_0)](#113-export-format-v2-q80)
- [12. Q8_0 Matmul Vectorization Optimization (2026-08-01)](#12-q80-matmul-vectorization-optimization-2026-08-01)
  - [12.1 Goal](#121-goal)
  - [12.2 Final Approach](#122-final-approach)
  - [12.3 KNC-Unavailable int8-Related Intrinsics (Compile/Link Failures)](#123-knc-unavailable-int8-related-intrinsics-compilelink-failures)
  - [12.4 Key Issues During Debugging](#124-key-issues-during-debugging)
  - [12.5 Q8_0 Inference "failed read" (Tokenizer Mismatch)](#125-q80-inference-failed-read-tokenizer-mismatch)
  - [12.6 Benchmark Results](#126-benchmark-results)
  - [12.7 Benchmark Notes](#127-benchmark-notes)
  - [12.8 Second Optimization: Multiple Accumulators Break the FMA Dependency Chain (2026-08-01)](#128-second-optimization-multiple-accumulators-break-the-fma-dependency-chain-2026-08-01)
  - [12.9 Third Optimization: FP32 matmul_imci with 4 Independent FMA Accumulators (2026-08-01)](#129-third-optimization-fp32-matmul_imci-with-4-independent-fma-accumulators-2026-08-01)
  - [12.10 Reliable Upload: scp Must Use -O (Legacy Protocol) (2026-08-01)](#1210-reliable-upload-scp-must-use--o-legacy-protocol-2026-08-01)

---

## 1. Project Background

Port [llama2.knc.c](https://github.com/) to run natively on an Intel Xeon Phi 7120P (Knights Corner, k1om architecture), rather than in MIC offload mode.

### Hardware Environment

| Item | Details |
|------|------|
| Model | Intel Xeon Phi 7120P (C0 stepping, 61 cores, ~1.24 GHz) |
| Memory | 15872 MB GDDR5, ECC enabled |
| Coprocessor OS | Linux 2.6.38.8+mpss3.8.6 (k1om, BusyBox) |
| IP | 192.168.1.100 |
| SSH | `ssh -i ~/.ssh/id_rsa_mic root@192.168.1.100` |
| Host OS | Windows 10 Pro for Workstations |

### Software Environment

| Item | Details |
|------|------|
| Compiler | Intel PSXE 2017 (17.0.0.109), `/Qmic` flag switches to k1om cross-compilation |
| Cross-compile host | WSL2 Ubuntu-22.04, calls the Windows-side `icl.exe` via WSL interop |
| k1om sysroot | `/opt/mpss/k1om-mpss-linux/` (glibc 2.21) |
| k1om MKL | `/opt/mpss/k1om-mkl/` (from PSXE 2017 `windows/mkl/lib/mic/`) |
| Runtime library | `libiomp5.so` (copied from PSXE 2017), MKL sequential |

### Key Architectural Differences

- **KNC (Knights Corner)** = IMCI/MVEX encoding, prefix `62`, different from AVX-512 EVEX encoding
- **KNL (Knights Landing)** = AVX-512 EVEX encoding, also uses the `62` prefix but with different semantics
- PSXE 2017 `/Qmic` → KNC/IMCI; `/Qmic-avx512` → KNL/AVX-512

---

## 2. Code Change Summary

### 2.1 Protecting Critical Code Paths

Since k1om does not support SSE/AVX/AVX2 instructions, the source needs conditional-compilation guards:

**run.c:**
1. `matmul_avx2` function (AVX2 intrinsics): wrapped with `#ifndef __MIC__` (lines 403-455)
2. `forward_cpu_bfp16` (AVX2 intrinsics): wrapped with `#if !defined(__INTEL_OFFLOAD) && !defined(__MIC__)` (line 1063)
3. MIC path macro mapping (lines 1300-1303): `#define forward_cpu_bfp16 forward_mic_bfp16`
4. MKL include change (lines 42-59): added `|| defined(__MIC__)` condition; under `__MIC__` include mkl.h directly

**bfp16.c:**
1. `matmul_bfp16_cpu` (AVX2): `#if !defined(__INTEL_OFFLOAD) && !defined(__MIC__)` (line 253)

**bfp16.h:**
1. Function declaration: same `#if !defined(__INTEL_OFFLOAD) && !defined(__MIC__)` (line 108)

### 2.2 Build/Link Scripts

| Script | Purpose | Status |
|------|------|------|
| `build_k1om_wsl_v2.sh` | Main build script (compile + link) | Done |
| `link_k1om.sh` | Link script (standalone) | Done |
| `upload_k1om.sh` | SCP upload to the 7120P | Done |
| `run_k1om.sh` | SSH run inference | Done |
| `build_novec.sh` | `-no-vec -no-simd` disable auto-vectorization | Done, still crashes |
| `build_scalar_mic.sh` | Fully scalar version (Python-replaced _mm512) | **Runs successfully** |
| `build_o0.sh` / `build_o1.sh` | Different optimization levels | O0/O1 both crash |
| `build_noomp.sh` | No-OpenMP version | Not tested |
| `build_hello.sh` | hello_k1om simple test | Passed |
| `build_align_test.sh` | Alignment test | Passed |
| `build_test_omp.sh` | OpenMP+_mm512 test | Passed |

---

## 3. Debugging Journey

### 3.1 Initial Approach: MIC Offload (abandoned)

- Compiled `llama2_mic.exe` (Windows + MIC offload)
- At runtime the `micx64.sys` driver caused a BSOD (`0x3B SYSTEM_SERVICE_EXCEPTION`)
- Decided to abandon offload and instead cross-compile a native k1om ELF using WSL + PSXE 2017

### 3.2 Setting Up the Cross-Compile Environment

- Call the Windows-side PSXE 2017 `icl.exe` from WSL2 Ubuntu-22.04 via interop
- Path conversion: `wslpath -w` → Windows UNC path (`\\wsl.localhost\...`)
- SSH key permissions: 0777 on Windows drvfs, must be copied to `/tmp/` and `chmod 600`
- PowerShell escaping: inside `bash -c "..."` the `$VAR` is expanded by PowerShell; use a script file instead

### 3.3 Fixing Compile Errors

1. **ICE (Internal Compiler Error) at run.c:1096** → guard AVX2 code with `#ifndef __MIC__`
2. **SSE/AVX2 instruction not supported** → add `#ifndef __MIC__` guard
3. **bfp16.c ICE at line 256** → extend the conditional compilation
4. **OpenMP symbols undefined** → link `libiomp5.so`
5. **MKL `libmkl_intel_thread.a` uses the GNU OpenMP API** → switch to `libmkl_sequential`
6. **Link output path issue** → copy `.o` to `/tmp/`, use the Windows UNC path

### 3.4 Debugging Runtime Crashes

#### Baseline tests (passed)

| Test program | Content | Result |
|----------|------|------|
| `hello_k1om` | `printf("Hello from KNC!\n")` | Passed — verifies the toolchain is correct |
| `test_align` | `_mm512_load_ps` aligned/unaligned | Passed — KNC supports unaligned 512-bit loads |
| `test_omp` | OpenMP + `_mm512_load_ps/add_ps/store_ps` | Passed — OpenMP + intrinsics combination works |

#### Crash tests (failed)

| Binary | Compile flags | Crash address | Crashing function | dmesg info |
|--------|---------|---------|---------|-----------|
| `llama2_k1om` | `-O2` | `ip:417b6f` (off +0x17b6f) | `forward_mic` | `general protection error:0` |
| `llama2_o0` | `-O0` | `ip:403790` (off +0x3790) | `rmsnorm` | `general protection error:0` |
| `llama2_novec` | `-O2 -no-vec -no-simd` | `ip:41282f` (off +0x1282f) | `forward_mic` | `general protection error:0` |
| `llama2_scalar` | `-O2` (scalarized) | — | — | **Runs successfully** |

#### Disassembly analysis

O2 and no-vec crash at the **exact same MVEX instruction sequence** inside `forward_mic`:
```
vmulps   (%rdi,%rax,4), %zmm?, %zmm?    // load and multiply
vbroadcastss 0x10(%rdi,%rax,4), %zmm?   // broadcast a scalar
vmulps   (%rbx,%rax,4), %zmm?, %zmm?    // ← crash point
```

O0 crashes at `vmovaps (%rax), %zmm0` inside `rmsnorm`.

---

## 4. Root Cause Analysis: Weight Memory Alignment

### 4.1 Hypothesis

**`vmovaps` on KNC requires 64-byte alignment. The weight data comes from `mmap()` memory-mapped file, and its start position is not 64-byte aligned, causing `_mm512_load_ps(weight + j)` to crash.**

### 4.2 Chain of Evidence

1. **The Config struct is 28 bytes** (7 `int`s):
   ```c
   typedef struct {
       int dim;           // 4
       int hidden_dim;    // 4
       int n_layers;      // 4
       int n_heads;       // 4
       int n_kv_heads;    // 4
       int vocab_size;    // 4
       int seq_len;       // 4
   } Config;              // = 28 bytes
   ```

2. **Weight start offset = 28 bytes**:
   ```c
   *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
   // mmap returns page-aligned memory (4096-byte aligned)
   float* weights_ptr = *data + sizeof(Config)/sizeof(float);
   // = *data + 7 floats = *data + 28 bytes
   // 28 % 64 = 28 ≠ 0  ← not 64-byte aligned!
   ```

3. **`_mm512_load_ps` → KNC `vmovaps`**: requires 64-byte alignment, otherwise GP fault (error:0)

4. **Why it crashes at different locations**:
   - O0: `rmsnorm` does a direct `vmovaps` on `weight` (unaligned) → crashes immediately
   - O2: the compiler may inline/reorder `rmsnorm`; the crash shows up inside `forward_mic`
   - no-vec: explicit intrinsics are unaffected by `-no-vec`; same crash location

5. **Why test_align passed**: the compiler likely generated unaligned load instructions such as `vmovdqu32` for the clearly-unaligned pointers in test_align, instead of `vmovaps`

6. **Why the scalar version passed**: scalar code does not use `_mm512_load_ps`, so it is naturally unaffected by alignment

### 4.3 Affected Locations

All `_mm512_load_ps(weight + offset)` calls (where weight comes from mmap) are affected:
- `rmsnorm(s_xb, x, w_rms_att_weight + l*dim, dim)` — **weight argument is unaligned**
- `matmul_imci(xout, x, w, n, d)` — **w argument is unaligned**

Activation buffers (`s_xb`, `x`, `s_q`, etc.) are allocated with `_mm_malloc(..., 64)`, which is 64-byte aligned, so they are unaffected.

---

## 5. Solutions

### Option A: Copy Weights to an Aligned Buffer in `read_checkpoint` (Recommended)

```c
void read_checkpoint(...) {
    // ... original mmap code ...

    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    
    // Check alignment
    if (((uintptr_t)weights_ptr & 63) != 0) {
        // Compute the weight size
        size_t weights_size = (*file_size - sizeof(Config)) / sizeof(float);
        // Allocate aligned memory and copy
        float* aligned_weights = (float*)_mm_malloc(weights_size * sizeof(float), 64);
        memcpy(aligned_weights, weights_ptr, weights_size * sizeof(float));
        // Update the pointer
        weights_ptr = aligned_weights;
        // Note: this buffer must be _mm_free'd in free_transformer
    }
    
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}
```

**Pros**: fixes all alignment problems at once, does not change the file format
**Cons**: extra memory overhead (~60 MB for stories15M)

### Option B: Use Unaligned Load Intrinsics

KNC has no `_mm512_loadu_ps`, but it has:
- `_mm512_loadunpacklo_ps` + `_mm512_loadunpackhi_ps` (used in pairs)
- `_mm512_extload_ps` (with swizzle control)

All loads of weight would need modification; large amount of work.

### Option C: Modify the Checkpoint File Format

Add padding after Config so weights are 64-byte aligned. But this breaks compatibility with existing checkpoint files.

### Option D: Manually Skip Alignment Bytes After the Config Struct

```c
float* weights_ptr = *data + sizeof(Config)/sizeof(float);
// Align to a 64-byte boundary
uintptr_t addr = (uintptr_t)weights_ptr;
uintptr_t aligned = (addr + 63) & ~63;
int skip_floats = (aligned - addr) / sizeof(float);
weights_ptr += skip_floats;
```

**Cons**: valid data gets skipped! Checkpoint files do not reserve padding.

---

## 6. Current Status and Benchmarks

### 6.1 Current Status

| Item | Status |
|------|------|
| k1om cross-compile toolchain | Verified (`hello_k1om` passed) |
| Source conditional-compilation changes | Done (run.c, bfp16.c, bfp16.h) |
| Weight alignment fix | **Done** — `read_checkpoint` detects and copies to a 64-byte aligned buffer |
| Vectorized version runs | **Works** — the `_mm512` intrinsics version runs correctly on the 7120P |
| Thread scaling | All 1-244 threads tested and passing, 71.3 tok/s @ 244 threads |
| Crash on exit | Fixed (`aligned_weights` is freed after `free_run_state`) |
| Final build approach | **Call PSXE 2017 icl.exe directly from PowerShell**, no longer depends on WSL interop |

### 6.1.1 Final Build Approach: Calling icl.exe Directly from PowerShell

Abandoned the WSL interop approach (drvfs permission issues + bash/PowerShell escaping conflicts); use PowerShell to call `icl.exe /Qmic` directly instead:

```powershell
cd "project_dir"

$ICL    = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\bin\intel64\icl.exe"
$MKL_INC = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\mkl\include"
$K1OM_INC = "C:\PROGRA~1\Intel\MPSS\k1om-mpss-linux\usr\include"
$MKL_LIB = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\mkl\lib\mic"
$K1OM_LIB = "C:\PROGRA~1\Intel\MPSS\k1om-mpss-linux\lib64"
$OMP_LIB = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\compiler\lib\mic"

# Compile
& $ICL /Qmic -c -I"$MKL_INC" -I"$K1OM_INC" -qopenmp -O2 run.c -o run_k1om.o
& $ICL /Qmic -c -I"$MKL_INC" -I"$K1OM_INC" -qopenmp -O2 bfp16.c -o bfp16_k1om.o

# Link (static MKL — avoid .so dependency)
& $ICL /Qmic run_k1om.o bfp16_k1om.o `
    "$MKL_LIB\libmkl_intel_lp64.a" "$MKL_LIB\libmkl_sequential.a" "$MKL_LIB\libmkl_core.a" `
    -L"$OMP_LIB" -liomp5 -L"$K1OM_LIB" -lpthread -lm -o llama2_k1om
```

**Key differences vs. the earlier WSL approach:**
- ✅ k1om sysroot uses the MPSS Windows installation path (`C:\Program Files\Intel\MPSS\`)
- ✅ MKL is statically linked (`.a`), no need to upload `.so` to the 7120P
- ✅ No drvfs permission issues, no PowerShell/bash escaping conflicts
- ⚠️ `libiomp5.so` still needs to be uploaded to the 7120P via WSL scp

### 6.2 Benchmark: stories110M.bin (12 layers, dim=768, hidden_dim=2048)

> The original KNC offload project reported: **24-29 tok/s** (fully offloaded to the MIC)
> Precision: float32, IMCI matmul (`-N 2`, default), non-BFP16 quantization
> Build: call `icl.exe /Qmic` directly from PowerShell, static MKL link (.a), MPSS k1om sysroot

| Threads | tok/s | Speedup | vs. Original Offload |
|--------|-------|--------|:------------:|
| 1 | 5.1 | 1.0x | 0.2x |
| 4 | 17.1 | 3.3x | 0.6x |
| 16 | **41.7** | 8.2x | **1.5x** |
| 32 | **59.6** | 11.7x | **2.2x** |
| 61 | **60.0** | 11.8x | **2.2x** |
| 122 | **66.0** | 12.9x | **2.5x** |
| **244** | **71.3** | **14.0x** | **2.6x** |

**Native k1om already surpasses the original offload approach at 16 threads, reaching 2.6x at 244 threads (71.3 vs 24-29 tok/s).** Thread scaling plateaus between 122→244 (+8%); the bottleneck is the "CPU-side" naive matmul (MKL sgemv call overhead + low efficiency for small matrices).

### 6.3 Benchmark: stories15M.bin (6 layers, dim=288, hidden_dim=768)

| Threads | tok/s |
|--------|-------|
| 1 | 29.84 |
| 4 | ~88 (estimated) |
| 16 | ~149 (estimated) |

### 6.4 Scalar Version Sample Output

```
Choosed naive matmul on CPU...
Choosed IMCI matmul on MIC...
Weights realigned: 0x7f3631ad001c -> 0x7f362e0d0040
...
hello Ollie, Bob, the turkey, was skipping in the woods...
249 tokens generated in 8.396 s, achieved tok/s: 29.656980
```

### Completed Fixes

1. **Weight alignment** (`#ifdef __MIC__`): `read_checkpoint` checks whether the mmap weight pointer is 64-byte aligned; if not, `_mm_malloc` + `memcpy`
2. **Cleanup order**: in `free_transformer`, `_mm_free(aligned_weights)` is placed after `free_run_state`
3. **Transformer struct**: added a `float* aligned_weights` field to track the aligned buffer

---

## 7. Technical Notes

### 7.1 Key KNC MVEX Intrinsic Mapping

| C Intrinsic | KNC Instruction | Alignment Requirement |
|-------------|---------|---------|
| `_mm512_load_ps` | `vmovaps` | 64 bytes |
| `_mm512_store_ps` | `vmovaps` | 64 bytes |
| `_mm512_set1_ps` | `vbroadcastss` | none (register operation) |
| `_mm512_setzero_ps` | `vpxor` / XOR | none |
| `_mm512_add_ps` | `vaddps` | none (register operation) |
| `_mm512_mul_ps` | `vmulps` | none (register operation) |
| `_mm512_fmadd_ps` | `vfmadd132ps`, etc. | none |
| `_mm512_reduce_add_ps` | horizontal add reduction sequence | none |
| `_mm512_exp_ps` | `vexp2ps` + adjustment (SVML) | none |

### 7.2 PSXE 2017 `/Qmic` Compiler Behavior

- `/Qmic` → targets k1om, defines `__MIC__`, outputs ELF
- MKL libraries: `libmkl_intel_lp64` + `libmkl_sequential` + `libmkl_core`
- The MVEX prefix byte `62` is the same as AVX-512 EVEX, but the encoding semantics of the following bytes differ
- KNC has 32 512-bit zmm registers (zmm0-zmm31)
- KNC has 8 mask registers (k0-k7)

### 7.3 WSL Interop Notes

- Convert paths with `wslpath -w` to Windows UNC paths
- Output `.o` files and binaries to `/tmp/` to avoid drvfs issues
- Copy the SSH key from drvfs to `/tmp/` and `chmod 600`
- Call WSL commands from PowerShell: `wsl -d Ubuntu-22.04 -e bash script.sh`
- Write long commands to `.sh` files to avoid PowerShell escaping problems

### 7.4 Known `_mm512` Usage Sites (run.c MIC Path)

1. **rmsnorm** (lines 321-331): `_mm512_set1_ps`, `_mm512_load_ps(x)`, `_mm512_load_ps(weight)`, `_mm512_mul_ps`, `_mm512_store_ps`
2. **softmax** (lines 343-351): `_mm512_load_ps`, `_mm512_max_ps`, `_mm512_reduce_max_ps`
3. **matmul_imci** (lines 463-500): `_mm512_setzero_ps`, `_mm512_load_ps`, `_mm512_mul_ps`, `_mm512_add_ps`, `_mm512_reduce_add_ps`
4. **forward_mic attention** (lines 666-675): same as above + attention score computation
5. **forward_mic weighted sum** (lines 702-742): `_mm512_set1_ps`, `_mm512_load_ps`, `_mm512_mul_ps`, `_mm512_add_ps`, `_mm512_store_ps`
6. **forward_mic residual connections** (lines 753-757, 793-798): `_mm512_load_ps`, `_mm512_add_ps`, `_mm512_store_ps`
7. **forward_mic SwiGLU** (lines 770-784): `_mm512_setzero_ps`, `_mm512_set1_ps`, `_mm512_load_ps`, `_mm512_sub_ps`, `_mm512_exp_ps`, `_mm512_add_ps`, `_mm512_div_ps`, `_mm512_mul_ps`, `_mm512_store_ps`

---

## 8. File Inventory

```
Project directory: C:\Users\Administrator\Desktop\llama.c\llama2.knc.c\llama2.knc.c-master\llama2.knc.c-master\

Core source:
  run.c          — Main inference engine (modified: conditional-compilation guards, scalar fallback version)
  bfp16.c        — BFP16 quantize/dequantize (modified: conditional-compilation guards)
  bfp16.h        — BFP16 header (modified: conditional-compilation guards)

Build scripts:
  build_k1om_wsl_v2.sh   — Main build script (WSL approach, obsolete)
  link_k1om.sh           — Link script (WSL approach, obsolete)
  upload_k1om.sh         — SCP upload script
  run_k1om.sh            — Remote run script
  build_novec.sh         — Vectorization-disabled build (for testing)
  build_scalar_mic.sh    — Fully scalar build (Python-replaced _mm512)
  build_noomp.sh         — No-OpenMP build (for testing)
  build_o0.sh            — -O0 build (for testing)
  build_o1.sh            — -O1 build (for testing)
  build_debug.sh         — -O0 -g debug build
  build_hello.sh         — hello_k1om test
  build_align_test.sh    — alignment test
  build_test_omp.sh      — OpenMP + intrinsics test
  build_and_bench.sh     — Full build + test (WSL approach, obsolete)
  build_fixed_k1om.sh    — Alignment-fix build (WSL approach, obsolete)
  benchmark.sh           — Benchmark script (WSL approach, obsolete)
  upload_and_bench.sh    — Upload + test (WSL approach)

  Recommended: call icl.exe directly from PowerShell (see 6.1.1); the WSL scripts are only used for SCP uploads. *

Analysis scripts:
  analyze_crash.sh       — symbol and disassembly analysis
  get_crash_info.sh      — crash info collection
  check_preproc.sh       — preprocessor directive checks

Test programs (C source):
  test_k1om.c            — assembly verification
  test_intrin.c          — extra intrinsic tests

Outputs / logs:
  crash_report.txt       — crash report (dmesg + exit code)
  sym_report.txt         — symbol table and disassembly
  scalar_build_log.txt   — scalar build log
  scalar_out.txt         — scalar version run output (successful)

Files in WSL /tmp/:
  run_scalar.c           — scalarized run.c (auto-generated by Python)
  llama2_k1om            — O2 build (crashes)
  llama2_o0              — O0 build (crashes)
  llama2_novec           — no-vec build (crashes)
  llama2_scalar          — scalar build (works!)
  hello_k1om             — test program (works)
  test_align             — test program (works)
  test_omp               — test program (works)

Files on 7120P /tmp/:
  llama2_k1om            — final version (static MKL + alignment fix, 543 KB)
  llama2_scalar          — scalar version (for debugging, 231 KB)
  stories15M.bin         — model weights (58 MB)
  stories110M.bin        — model weights (419 MB)
  tokenizer.bin          — tokenizer (424 KB)
  libiomp5.so            — OpenMP runtime (1.3 MB)
  hello_k1om             — toolchain verification program
  test_align, test_omp   — debug test programs
```

---

## 9. Quick Command Reference

```powershell
# ========== Compile (direct PowerShell call) ==========
cd "c:\Users\Administrator\Desktop\llama.c\llama2.knc.c\llama2.knc.c-master\llama2.knc.c-master"

$ICL    = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\bin\intel64\icl.exe"
$MKL_INC = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\mkl\include"
$K1OM_INC = "C:\PROGRA~1\Intel\MPSS\k1om-mpss-linux\usr\include"
$MKL_LIB = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\mkl\lib\mic"
$K1OM_LIB = "C:\PROGRA~1\Intel\MPSS\k1om-mpss-linux\lib64"
$OMP_LIB = "C:\PROGRA~2\INTELS~1\COMPIL~1.109\windows\compiler\lib\mic"

& $ICL /Qmic -c -I"$MKL_INC" -I"$K1OM_INC" -qopenmp -O2 run.c -o run_k1om.o
& $ICL /Qmic -c -I"$MKL_INC" -I"$K1OM_INC" -qopenmp -O2 bfp16.c -o bfp16_k1om.o
& $ICL /Qmic run_k1om.o bfp16_k1om.o "$MKL_LIB\libmkl_intel_lp64.a" "$MKL_LIB\libmkl_sequential.a" "$MKL_LIB\libmkl_core.a" -L"$OMP_LIB" -liomp5 -L"$K1OM_LIB" -lpthread -lm -o llama2_k1om

# ========== Upload (WSL scp) ==========
wsl -d Ubuntu-22.04 -e bash -c "cp /mnt/c/Users/Administrator/.ssh/id_rsa_mic /tmp/k; chmod 600 /tmp/k; scp -i /tmp/k -o StrictHostKeyChecking=no '/mnt/c/Users/Administrator/Desktop/llama.c/llama2.knc.c/llama2.knc.c-master/llama2.knc.c-master/llama2_k1om' root@192.168.1.100:/tmp/; rm -f /tmp/k"

# First deployment also needs:
wsl -d Ubuntu-22.04 -e bash -c "cp /mnt/c/Users/Administrator/.ssh/id_rsa_mic /tmp/k; chmod 600 /tmp/k; scp -i /tmp/k -o StrictHostKeyChecking=no /opt/mpss/k1om-mkl/lib/libiomp5.so root@192.168.1.100:/tmp/; scp -i /tmp/k -o StrictHostKeyChecking=no '/mnt/c/Users/Administrator/Desktop/llama.c/llama2.knc.c/llama2.knc.c-master/llama2.knc.c-master/tokenizer.bin' root@192.168.1.100:/tmp/; scp -i /tmp/k -o StrictHostKeyChecking=no '/mnt/c/Users/Administrator/Desktop/llama.c/llama2.knc.c/llama2.knc.c-master/llama2.knc.c-master/stories110M.bin' root@192.168.1.100:/tmp/; rm -f /tmp/k"

# ========== SSH run ==========
ssh -i $env:USERPROFILE\.ssh\id_rsa_mic -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@192.168.1.100 "cd /tmp; LD_LIBRARY_PATH=/tmp OMP_NUM_THREADS=244 ./llama2_k1om stories110M.bin -i 'Once upon a time'"

# ========== Debugging ==========
# SSH directly to the 7120P
ssh -i $env:USERPROFILE\.ssh\id_rsa_mic -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@192.168.1.100

# View the crash log
ssh ... root@192.168.1.100 'dmesg | tail -20'

# Check files on the 7120P
ssh ... root@192.168.1.100 'ls -lh /tmp/llama2* /tmp/stories*'
```

---

## 10. Matmul Precision Modes

Currently uses **float32 IMCI** (`-N 2`, default), not BFP16 quantization.

| Value | Mode | Precision | Purpose |
|:--:|------|------|------|
| 0 | naive | float32 | Pure CPU serial multiplication |
| 1 | MKL | float32 | MKL `sgemv` call |
| 2 | **IMCI** | **float32** | **Currently used — manual loop with `_mm512` intrinsics** |
| 3 | BFP16 | bf16 quantized weights | int16 weight + scale, dequantized at inference, saves 50% memory |

BFP16 mode is enabled with `-N 3`. The idea: `generate()` calls `bfp16_packW()` to quantize the float32 weights
into `(int16, scale)` pairs; `matmul_bfp16_mic` dequantizes first and then computes.

### 10.1 BFP16 Test Results (Unusable)

| Metric | IMCI float32 | BFP16 |
|------|:---------:|:-----:|
| tok/s (110M, OMP=1) | 5.1 | 3.0 |
| tok/s (110M, OMP=244) | 71.3 | 30.2 |
| Memory (110M) | 417.8 MB | 235.0 MB (56%) |
| Output quality | Normal | **Garbled / unusable** |

**Why BFP16 is unusable:**

1. **Quantization happens at runtime (7120P side)**: `bfp16_quantize()` is called inside `generate()`.
   The float32 weights are first loaded into the 7120P's memory and then quantized. The 7120P must hold both fp32 + BFP16 weights at once
   (110M needs 417+235=652MB), so no memory is saved. For llama2-7b this means 26GB+7GB=33GB,
   far exceeding the 7120P's 16GB.

2. **Very poor output quality**: BFP16 mode never appears in the original project's README and is an unvalidated code path.
   In `matmul_bfp16_mic`, `_mm512_load_ps(w_fp32)` loads a local array from the stack, which may
   hide a stack alignment issue.

3. **Actually slower (42%)**: dequantization is performed block-by-block in the inner loop, with significant extra overhead.

### 10.2 Int8 Quantization (Q8_0, runq.c)

The original project provides a complete int8 quantization scheme (`runq.c`), independent from the `run.c` we currently use:

| Feature | BFP16 (run.c -N 3) | Q8_0 Int8 (runq.c) |
|------|:-----:|:-----:|
| Quantization timing | **Runtime (7120P side)** | **Export time (PC side)** |
| Weight size (7B) | ~7GB (theoretical, but infeasible) | ~6.7GB |
| Needs fp32 loaded simultaneously? | **Yes (2x memory)** | **No** |
| Speedup | Slower (0.4x) | ~3x (README) |
| Format | int16/block + float scale | int8/group + float scale |
| Group size | 16 (BFP16_BLOCK_SIZE) | 64 (GS) |

**Q8_0 quantization method:**
- Symmetric quantization: `q = round(w / scale)`, `scale = max(abs(w)) / 127.0`
- One scale shared per 64 elements
- Export: `python export.py llama2_7b_q80.bin --version 2 --hf meta-llama/Llama-2-7b-hf`
- Inference: `runq.c` compiled and run independently

**Porting runq.c to k1om requires:**
- `runq.c`'s `matmul()` uses AVX2 `_mm256` intrinsics for the int8 dot product
- Must be changed to KNC `_mm512` intrinsics
- Idea: use `_mm512_load_ps` + `_mm512_fmadd_ps` to dequantize and compute (similar to `matmul_imci`)
  rather than doing the int8 dot product directly (KNC has no `_mm512_dpbusd_epi32` VNNI instruction)

### 10.3 llama2-7b Deployment Plan

| Plan | Weight size | 7120P memory | Feasibility |
|------|:------:|:--------:|:----:|
| fp32 IMCI (run.c) | 26 GB | Exceeds (16GB) | Infeasible |
| BFP16 runtime quantization (run.c -N 3) | needs 26+7=33GB at once | Exceeds (16GB) | Infeasible |
| **Q8_0 int8 (runq.c → k1om)** | **6.7 GB** | **Fits** | **Recommended** |
| Q8_0 int8 (export + existing run.c) | 6.7 GB | Fits | Needs an int8 matmul |

**Recommended path:** export Q8_0 int8 format with `export.py --version 2`, then
port `runq.c`'s matmul to KNC `_mm512` intrinsics (or extend the existing k1om path in `run.c`). *

---

## 11. llama2-7b Download and Export

### 11.1 Prerequisites

- Meta Llama 2 requires HuggingFace authentication (apply for access: https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Python dependencies: `torch`, `transformers`, `numpy`, `tqdm`
- Install: `pip install torch numpy tqdm transformers huggingface_hub`

### 11.2 Export Commands

```bash
# fp32 (26GB, for run.c)
python export.py llama2_7b.bin --version 1 --hf meta-llama/Llama-2-7b-hf

# Q8_0 int8 (6.7GB, for runq.c or a k1om int8 matmul)
python export.py llama2_7b_q80.bin --version 2 --hf meta-llama/Llama-2-7b-hf
```

### 11.3 Export Format (v2 Q8_0)

.bin header (256 bytes):
- magic: 0x616b3432
- version: 2
- dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
- shared_classifier (1 or 0)
- group_size (GS, default 64)

Weight layout: each quantized tensor = int8 data[] + float scale[groups] interleaved.

---

## 12. Q8_0 Matmul Vectorization Optimization (2026-08-01)

### 12.1 Goal

Change the Q8_0 quantized inference path `matmul_q80` (`src/run.c`) from pure scalar to KNC IMCI vectorization, to speed up Mistral-7B Q8_0 inference.

### 12.2 Final Approach

- Dequantization must keep auto-vectorization: KNC icc's vector-conversion intrinsics for explicit int8 loads (`vloadunpacklps/hps`) are unavailable (see 12.3)
- **Approach**: use `#pragma omp simd` to dequantize int8→fp32 into a 64-byte aligned local buffer `wf[64]`, then use 4 handwritten `_mm512_fmadd_ps` for the multiply-accumulate (each 64-element chunk split into four 16-element chunks)
- `_mm512_load_ps` requires 64-byte alignment: at the call sites `s_xb`/`s_hb`/`x` are all allocated with `_mm_malloc(..., 64)`, so it is safe

```c
void matmul_q80(float* xout, float* x, int8_t* wq, float* ws, int n, int d) {
    int nn = n / GS_Q80 * GS_Q80;  // largest multiple of 64
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        const int8_t* wrow = wq + (size_t)i * n;
        const float* srow = ws + ((size_t)i * n) / GS_Q80;
        float wf[64] __attribute__((aligned(64)));
        // 4 independent accumulators (one per 16-element chunk) break the FMA dependency chain, improving ILP (see 12.8)
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();
        int j = 0;
        for (; j < nn; j += GS_Q80) {
            const float sc = srow[j / GS_Q80];
            #pragma omp simd
            for (int k = 0; k < GS_Q80; k++) wf[k] = (float)wrow[j + k] * sc;
            sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf),      _mm512_load_ps(x + j),      sum0);
            sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16), _mm512_load_ps(x + j + 16), sum1);
            sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32), _mm512_load_ps(x + j + 32), sum2);
            sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48), _mm512_load_ps(x + j + 48), sum3);
        }
        __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        float val = _mm512_reduce_add_ps(sum);
        #pragma novector
        for (; j < n; j++) val += (float)wrow[j] * srow[j / GS_Q80] * x[j];
        xout[i] = val;
    }
}
```

Correctness was verified with `test/test_q80_intrin.c` (serial / omp-simd / 8-thread parallel+simd variants all exactly match the volatile scalar ground truth, max_abs_err=0).

### 12.3 KNC-Unavailable int8-Related Intrinsics (Compile/Link Failures)

| intrinsic | Problem |
|-----------|------|
| `_mm512_cvtepi8_ps` / `_mm512_cvtepu8_ps` | declared implicitly |
| `_mm512_maddubs_epi16` / `_mm512_madd_epi16` | declared implicitly |
| `_mm512_castsi512_si128` | declared implicitly |
| `_mm512_cvtepi32_ps` | undefined reference (explicitly linking libirc.a does not help) |
| `_mm512_loadu_ps` | undefined reference |

→ Abandoned the integer dot-product (VNNI-like) approach, switched to "auto-vectorized dequantization + handwritten FMA".

### 12.4 Key Issues During Debugging

1. **test_q80_intrin.c `ws` allocation missing `sizeof(float)`**: `_mm_malloc(d * (n/GS_Q80), 64)` allocated only 1/4 of the needed size, causing the following buffers (r0/r1/r_ref) to overlap with `ws`; the fill then wrote out of bounds and corrupted heap metadata → a variety of bizarre symptoms (28809 out-of-range values, `free(): invalid next size`, scalar/vec results out of sync). Fix: add `* sizeof(float)`. This was the key breakthrough in this diagnosis.
2. **KNC icc's LCG data-fill bug**: the old test used an LCG with `% 251` + `/ 1e9f`; KNC icc computes it with a floating-point reciprocal (`vrcp23ps`), giving completely wrong results when the seed is ~2^32. Switched to a simple power-of-2 data pattern.
3. **False alarm about a compiler bug, clarified**: before the WS overlap bug was fixed, `vloadunpacklps` appeared to load misaligned (`-32(%rbx)` negative offset), and it was suspected that KNC icc had an int8 auto-vectorization bug; after the WS fix, all `#pragma omp simd` variants report ALL OK — there is **actually no compiler bug**.
4. **`matmul_*` inlined into main causing compile confusion**: the old test (no noinline) produced code with suspected value mixing in main. Fixed by adding `__attribute__((noinline))`.

### 12.5 Q8_0 Inference "failed read" (Tokenizer Mismatch)

**Symptom**: `mistral7b_q80.bin` prints `failed read` after "building tokenizer..." and exits (exit 1), before any matmul runs.

**Root cause**: the model is **Mistral-7B-Instruct-v0.3** (`vocab_size=32768`, the 6th header int is -32768 meaning unshared weights), while `/tmp/tokenizer.bin` is the llama2 **32000-token** version (433869 bytes). `build_tokenizer` loops `fread` 32768 times according to `config.vocab_size`; at read #32001 it hits EOF → "failed read".

**Header verification** (read the first 40 bytes remotely with `dd + od -i`):
```
4096 0 14336 0 32 0 32 0 8 0 -32768 -1 -32768 0 16385 0
→ dim=4096, hidden_dim=14336, n_layers=32, n_heads=32, n_kv_heads=8,
  vocab_size=(-32768)→32768, seq_len=32768
```

**Fix**: the tokenizer.bin in the project's `Mistral-7B-Instruct-v0.3` directory has 32768 tokens (444974 bytes, max_token_length=25); upload it as `/tmp/tokenizer_mistral.bin` and specify it explicitly with `-z /tmp/tokenizer_mistral.bin` at runtime.

> Note: `upload_k1om.ps1` overwrites `/tmp/tokenizer.bin` with the llama2 32000-token version on every run; Mistral models must use `-z` to explicitly specify the 32768-token tokenizer, otherwise "failed read".

### 12.6 Benchmark Results

**Q8_0 path (Mistral-7B, 16 tokens, clean serial benchmark)**

| Threads | Old scalar (tok/s) | New vectorized (tok/s) | Improvement |
|------|:---:|:---:|:---:|
| 244 | 6.79 | **10.52** | **+55%** |
| 122 | 5.61 | 6.08 | +8% |
| 60 | 3.96 | 4.39 | +11% |
| 32 | 2.15 | 3.19 | +48% |
| 16 | 1.08 | 1.62 | +50% |
| 4 | 0.32 | 0.51 | +59% |

**FP32 path (stories110M @61 threads)**: 58.7 → **61.95 tok/s (+5.5%)**, generated text is semantically correct.

### 12.7 Benchmark Notes

- **Benchmarks must be serial**: multiple llama2_k1om processes contending for the CPU drag tok/s way down (244 threads 10.4 → 3.8). Before benchmarking, run `pkill -9 llama2_k1om` to confirm no leftover processes, then run each thread count.
- The segfault on exit (exit 139) is the known libiomp5/glibc 2.21 exit crash from the README; it does not affect inference results or tok/s stats.

### 12.8 Second Optimization: Multiple Accumulators Break the FMA Dependency Chain (2026-08-01)

#### Motivation

In the single-accumulator version, the 4 FMAs serially accumulate into the same `__m512 sum`, forming a 4-deep dependency chain. KNC FMA latency is ~4-5 cycles, and single-core ILP is limited by this. Switching to **4 independent `__m512` accumulators** (one per 16-element chunk) makes all 4 chains fully independent and issuable in parallel, theoretically 4x ILP.

#### Correctness Verification

`test/test_q80_intrin.c` adds a `matmul_q80_vec4` variant, compared three ways against the volatile scalar ground truth: **max_abs_err=0, ALL OK**.

#### Benchmark Results (Mistral-7B Q8_0, 16 tokens, clean serial benchmark)

| Threads | Single accumulator (tok/s) | Multiple accumulators (tok/s) | Difference |
|------|:---:|:---:|:---:|
| 244 | 10.52 | 10.40 | -1.2% |
| 122 | 6.08 | 6.25 | +2.8% |
| 60 | 4.39 | 4.25 | -3.2% |
| 32 | 3.19 | 3.25 | +1.8% |
| 16 | 1.62 | **1.88** | **+15.9%** |
| 4 | 0.51 | 0.52 | +1.6% |

**Supplement** (-n 48 longer sequences, reducing startup overhead share): 60→3.999, 32→2.957, 16→1.640.

#### Conclusion

- **Low thread counts clearly benefit (+16% at 16 threads)**: the single-core FMA dependency chain is the bottleneck; the 4-accumulator ILP directly translates into throughput.
- **High thread counts (60+) are in the noise range (±3%)**: the bottleneck shifts to inter-thread scheduling / memory bandwidth, and single-core ILP gains are no longer visible.
- Overall gains are modest but side-effect free (0 error rate, small code change); merged into the official `src/run.c`.

### 12.9 Third Optimization: FP32 `matmul_imci` with 4 Independent FMA Accumulators (2026-08-01)

#### Motivation

The FP32 path (`matmul_imci`) previously only had basic IMCI vectorization (4-way parallel 512-bit loads + separate `mul`/`add`), with two problems:
1. **No FMA**: `_mm512_mul_ps` + `_mm512_add_ps` are two instructions; KNC natively supports `_mm512_fmadd_ps` which does it in one, halving the instruction count
2. **Single accumulator**: the 4 chunks serially accumulate into the same `sum_vec`, forming an FMA dependency chain that limits ILP (same problem as matmul_q80 before the multi-accumulator change)

The change is exactly the same as `matmul_q80`: 4 independent `__m512` accumulators (one per 16-element chunk) + `_mm512_fmadd_ps`, then `_mm512_add_ps` to merge the 4 partial sums followed by a horizontal reduction.

#### Benchmark Results (stories110M FP32, 50 tokens, clean serial benchmark, both binaries compared in the same session)

| Threads | Old mul+add single accumulator (tok/s) | New 4-FMA accumulator (tok/s) | Improvement |
|------|:---:|:---:|:---:|
| 244 | 80.07 | 79.03 | -1.3% (noise) |
| 122 | 79.80 | 82.63 | +3.5% |
| 61 | 74.13 | 78.65 | +6.1% |
| 32 | 69.41 | 75.50 | +8.8% |
| 16 | 47.02 / 47.71 | 63.23 / 63.97 | **+33%** |
| 4 | 17.87 / 17.77 | 27.67 / 27.86 | **+56%** |

#### Conclusion

- **Low thread counts benefit greatly**: +33% at 16 threads, +56% at 4 threads — the single-core FMA dependency chain is the main bottleneck, and the 4 independent accumulators directly translate into throughput. This matches the Q8_0 multi-accumulator gains (and is more pronounced, since FP32 has no quantization/dequantization overhead).
- **High thread counts (61+) improve slightly (+3~6%)**; 244 threads is flat (noise range).
- Generated text is semantically normal; the change is confined to the `matmul_imci` function body; merged into `src/run.c`.

### 12.10 Reliable Upload: scp Must Use `-O` (Legacy Protocol) (2026-08-01)

#### Problem

Windows OpenSSH `scp.exe` **uses the SFTP subsystem by default**, and the 7120P's BusyBox **has no SFTP**, causing:

```
$ scp -i ~/.ssh/id_rsa_mic build\llama2_k1om_new root@192.168.1.100:/tmp/
C:\Windows\System32\OpenSSH\scp.exe: Connection closed   (exit 255)
```

The failure is not network/key related, but a protocol mismatch. The new Windows `scp` defaults to `sftp`, while the remote BusyBox only supports the legacy scp protocol.

#### Verified Stable Command (tested)

```powershell
scp -O -i "$env:USERPROFILE\.ssh\id_rsa_mic" -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
    .\build\llama2_k1om_new root@192.168.1.100:/tmp/
```

- **`-O` (legacy protocol) is mandatory**: without it you get "Connection closed" (measured exit 255)
- On success the output is `100% 595KB 15.7MB/s 00:00`, exit code 0
- Integrity check: local and remote MD5 match
  - Local: `Get-FileHash .\build\llama2_k1om_new -Algorithm MD5`
  - Remote: `ssh ... "md5sum /tmp/llama2_k1om_new"`

#### `chmod +x` is Mandatory After Upload

scp preserves source file permissions by default; files uploaded from Windows are usually `-rw-r--r--` on the remote side (not executable), and running them directly gives `Permission denied`:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_rsa_mic" root@192.168.1.100 "chmod +x /tmp/llama2_k1om_new"
```

#### Summary

| Operation | Command key point |
|------|---------|
| Upload | `scp -O ...` (`-O` is mandatory, otherwise SFTP → Connection closed) |
| Permissions | `chmod +x /tmp/<bin>` after upload (required under BusyBox) |
| Verify | compare remote `md5sum` with local `Get-FileHash -Algorithm MD5` |
| Reference script | `deploy/upload_k1om.ps1` already includes `scp -O` (see the comment at the top of the script) |

### 12.11 Fourth Optimization Attempt: RoPE Lookup Table (2026-08-01, **no benefit, reverted**)

#### What Changed

In the RoPE loops of `forward_mic_q80` and `forward_mic`, `fcr/fci` depend only on `head_dim` (only head_size values, e.g. 128) and the fixed `pos`. The original code calls `powf/cosf/sinf` inside the **per-element loop** (Mistral: 32 layers × ~2048 cos/sin pairs per token ≈ **200k libm calls**).

Change: precompute two lookup tables `rope_cos[head_size]` / `rope_sin[head_size]` at the start of the forward function (only 2×head_size trig calls per token), then look them up directly in the loop:

```c
// RoPE lookup table: precompute once per forward call (shared by all layers)
float rope_cos[head_size];
float rope_sin[head_size];
for (int hd = 0; hd < head_size; hd++) {
    float freq = 1.0f / powf(10000.0f, hd / (float)head_size);
    float val = pos * freq;
    rope_cos[hd] = cosf(val);
    rope_sin[hd] = sinf(val);
}
// ... inside the RoPE loop: float fcr = rope_cos[i % head_size]; float fci = rope_sin[i % head_size];
```

#### Benchmark Results (Mistral-7B Q8_0, 16 tokens, same-session comparison with the multi-accumulator version)

| Threads | Multi-accumulator (tok/s) | RoPE lookup table (tok/s) | Difference |
|------|:---:|:---:|:---:|
| 244 | 10.40 | 10.30 | -1.0% |
| 122 | 6.11 | 6.14 | +0.5% |
| 60 | 4.33 | 4.35 | +0.5% |
| 32 | 3.19 | 2.98 | -6.6% (noise) |
| 16 | 1.75 | 1.86 | +6.3% (noise) |

#### Conclusion (Key Takeaway)

**Overall tok/s is unchanged, all within the noise range.** Reasons:

1. **The RoPE loop is already parallelized by `#pragma omp parallel for simd`** — 200k libm calls spread over 244 threads is only ~800 calls per thread; the overhead was already tiny
2. **KNC icc may have already vectorized `cosf/sinf` in the loop via SVML** (k1om has hardware support like `vexp2ps`), further diluting any gain
3. **matmul is the absolute dominant cost**: each Mistral layer's matmul is ~140M FMAs; the RoPE trig computation share is <1%

**Takeaway**: don't optimize `omp parallel for simd`-parallelized loops based on "high call count" intuition — parallelization already flattens the scalar overhead. Measure the hot-spot share before deciding what to optimize.

**Final decision (user)**: do not use the RoPE lookup table; **reverted** the related changes in `forward_mic_q80` and `forward_mic`; the code is back to the multi-accumulator version (equivalent to the benchmarked `llama2_k1om_new`), and `build/llama2_k1om` was rebuilt (595.4KB, compiles cleanly). This document keeps the record as a "failed optimization" reference.

### 12.12 Fifth Optimization: attention score/value Multi-Accumulator + FMA (2026-08-01, **+7~13% at low thread counts**)

#### What Changed

The two loops in the attention section have the same dependency-chain problem as matmul before the optimization; fixed with the same pattern:

1. **score computation** (`forward_mic_q80` and `forward_mic`): was a **single accumulator** `_mm512_mul_ps` + `_mm512_add_ps` (128 elements → 8 16-element chunks all serially accumulated). Changed to **4 independent accumulators + `_mm512_fmadd_ps`** (4 parallel FMAs per round, 2 rounds), then a final merge reduce.
2. **value weighted accumulation** (t=1..pos loop): was two instructions `mul` + `add`; merged into a single `_mm512_fmadd_ps` (halved instruction count).
3. **`att[t] = score / sqrtf(head_size)`**: `sqrtf` is called repeatedly per (head, t); changed to precompute `inv_sqrt_head = 1.0f / sqrtf(head_size)` at the start of the function and multiply in the loop.

#### Benchmark Results (Mistral-7B Q8_0, 16 tokens, same-session comparison with the multi-accumulator version)

| Threads | Multi-accumulator (tok/s) | +attention optimization (tok/s) | Difference |
|------|:---:|:---:|:---:|
| 244 | 10.18 | 10.09 | -0.9% (noise) |
| 122 | 6.38 | 6.28 | -1.6% (noise) |
| 60 | 4.13 | 4.38 | +6.1% |
| 32 | 2.99 | 3.20 | +6.8% |
| 16 | 1.67 | **1.89** | **+13.4%** |

> Note: during this benchmark the system load was high (uptime load ~15-35); high-thread data contains noise. The low-thread gain trend is clear.

#### Conclusion

- **Low thread counts clearly benefit** (+13% at 16 threads, +7% at 32) — the serial accumulation chain of attention scores is a real single-core bottleneck; 4 accumulators directly raise ILP, exactly matching the matmul multi-accumulator gain pattern.
- **High thread counts are flat** (bottleneck is matmul throughput and thread scheduling).
- Generated text is semantically normal. The change was merged into `src/run.c`; the build artifact is `build/llama2_k1om_attn` (595.2KB).

### 12.13 Sixth Optimization Attempt: -O3 + -fp-model fast=2 Compile Flags (2026-08-01, **no benefit, reverted to default**)

#### What Changed

[build_k1om.ps1](scripts/build_k1om.ps1) gained a `-Fast` switch that appends `-fp-model fast=2` (relaxed FP semantics, approximate math functions). Test combination: `-O3 -Fast`.

#### Benchmark Results (Mistral-7B Q8_0, 16 tokens)

The first full pass (bench8) was polluted by system load (load 15-44; o3fast happened to run in the high-load window), so the data is untrustworthy; a clean 32/16-thread comparison was then done twice each (bench8b):

| Threads | attn (O2) | o3fast (O3+fast=2) |
|------|:---:|:---:|
| 32 | 2.97 / 2.88 | 2.87 / 2.90 |
| 16 | 1.87 / 1.87 | 1.80 / 1.87 |

**Conclusion: no benefit**, both sets are flat within the noise range.

#### Conclusion (Takeaway)

- **Handwritten-intrinsics matmul/attention is already optimal at -O2** — `-O3` mainly means more aggressive loop unrolling / instruction scheduling, with limited gain for handwritten SIMD kernels
- **`-fp-model fast=2` only affects libm math functions** (expf/cosf/sinf, etc.), and these have been proven not to be the bottleneck (see the 12.11 RoPE experiment); the approximations for softmax/sigmoid also show no visible difference in inference quality
- **Final decision (user)**: the default build reverts to `-O2` (consistent with the verified version); the build script **keeps the `-Fast` switch** for future experiments; this document records it as a "failed optimization" reference.

### 12.14 Seventh Optimization Attempt: softmax Vectorization (`_mm512_exp_ps`, 2026-08-01, **+2.9% at high thread counts, kept**)

#### Motivation

Authoritative verification of intrinsics (see the 12.3 revision + Chapter 11 of the tech summary) found that **`_mm512_exp_ps` compiles and links cleanly** (KNC has the hardware `vexp2ps`), while the exp section of softmax is still scalar `expf` + `omp parallel for simd`. The original code even has a commented-out vectorized exp (the author worried about an "alignment problem with att").

#### What Changed ([run.c](src/run.c) MIC softmax)

- Verified that `att`/`logits` are both allocated with `_mm_malloc(..., 64)`, so **64B alignment is guaranteed**; the alignment problem the original author worried about does not actually exist
- Enabled vectorized exp+sum with `_mm512_exp_ps` (`_mm512_sub_ps` → `_mm512_exp_ps` → `_mm512_reduce_add_ps`); the normalize section is vectorized as well (`_mm512_div_ps`)
- Added a `(uintptr_t)x & 63` alignment guard: falls back to the scalar path when x is unaligned (defensive for future callers)
- Also fixed the out-of-bounds hazard in the original normalize loop `for (i=0;i<size;i+=4)`

#### Benchmark Results (stories110M FP32, 128 tokens, -t 0.9 -s 42, mean of 3 runs)

| Threads | Baseline (llama2_k1om_attn) | exp512 | Difference |
|------|:---:|:---:|:---:|
| 4 | 26.45 | 26.80 | +1.3% |
| 16 | 59.85 | 60.22 | +0.6% |
| 61 | 66.82 | 67.03 | +0.3% |
| **244** | **79.93** | **82.26** | **+2.9%** |

#### Conclusion

- **No gain at low thread counts** (+0.3~1.3%, noise range): softmax is a small share of total time, and the multi-threaded parallelism of the original `omp parallel for simd` is offset by the sequential vectorization
- **+2.9% at 244 threads** (real gain): at high thread counts the outer attention `omp parallel for (h)` already occupies all threads, and the inner nested parallel region is serialized → the original expf section effectively runs scalar, exposing the `vexp2ps` hardware advantage
- Numerical correctness: the 4-thread output matches the baseline character-for-character; at 16/61/244 threads the different float summation order causes sampling paths to diverge (normal at temperature 0.9); all generated text is valid
- **Final decision (user)**: keep the exp512 implementation (safe with the alignment guard, and it pays off at high thread counts)

#### Takeaways

- `_mm512_exp_ps` works end-to-end and is correct → future high-precision exp vectorization (e.g. sigmoid) can reuse it directly
- "Code the original author commented out" is not necessarily wrong — verify the premise (alignment) yourself before deciding whether to enable it

#### Follow-up Fix (2026-08-02, Qwen2.5 k_bias debugging)

The 12.14 vectorized max-reduction had a latent bug that produced NaN: `__m512 max_val_load = _mm512_load_ps(x)` loads 16 elements even when `size < 16` (pos 0..14, since `size = pos+1`). The stale garbage in `x[size..15]` (NaN left over from a previous softmax) corrupted `max_val` via `_mm512_reduce_max_ps`, turning the whole softmax output NaN. Qwen's k_bias (±300) changed the attention score magnitudes and deterministically exposed this. **Fix**: `max_val` is now computed only over the valid `size` elements (vector reduce when `size_aligned >= 16`, else scalar). Additionally the exp input is clamped to `[-80, 0]` because with k_bias the score range reaches ±324 → `exp(x-max)` input -324 → 2^-467 is below KNC `vexp2ps`'s subnormal range, which returns NaN instead of 0. See `docs/QWEN_CN_STATUS.md` for details.