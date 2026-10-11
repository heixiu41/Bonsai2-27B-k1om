# llama2.knc.c MIC Offload Compilation Log

> **2026-08-01 update**: The MIC offload approach has been abandoned (BSOD 0x3B driver conflict). The current approach is **native k1om cross-compilation with PowerShell + icc.exe (intel64_mic)**, see section 12 of `DEBUG_LOG.md`.
>
> The following is the original record of the MIC offload compilation attempts.

---
## Goal

Compile the [llama2.knc.c](https://github.com/nicholas-lly/llama2.knc.c) project into a MIC offload version usable on Xeon Phi 7120P (Knights Corner, k1om), so that BFP16 quantized inference is accelerated on the coprocessor.

## Environment

| Component | Version |
|---|---|
| Operating System | Windows 11 |
| Compiler | Intel C++ Compiler 17.0.0.109 (PSXE 2017) |
| MSVC | VS 2022 Community, MSVC 14.51.36231 |
| Windows SDK | 10.0.26100.0 |
| MKL | Bundled with PSXE 2017 |

> Note: PSXE 2017 is the last version supporting KNC (Knights Corner). PSXE 2018/19/20 have removed KNC support.

### Why VS 2015 + MSBuild was not used

PSXE 2017 supports VS 2012-2015 integration. VS 2015 Community was installed but lacks the VC++ compiler component (`cl.exe` does not exist), so MSBuild cannot be used. Instead, a combination of command-line `icl.exe` with VS 2022 MSVC header files/libraries was adopted.

## Modified Files

### 1. `bfp16.h` - add `size_t` declaration

```c
// Before
#include <stdint.h>

// After
#include <stddef.h>   // provides size_t
#include <stdint.h>
```

**Reason:** The MIC sub-compiler does not inherit the implicit header files of the host MSVC, so `size_t` must be explicitly pulled in via `<stddef.h>`.

### 2. `run.c` and `bfp16.c` - fix macro definitions (completed in a previous session)

```c
#elif defined(__INTEL_OFFLOAD) && defined (_WINHOST)
#define TARGET_MIC_ATTR __declspec(target(mic))
#define TARGET_MIC_PUSH  /* inline #pragma offload_attribute(push,target(mic)) */
#define TARGET_MIC_POP   /* inline #pragma offload_attribute(pop) */
```

### 3. `llama2.knc.c.run.vcxproj` / `llama2.knc.c.runq.vcxproj`

- Changed `<WindowsTargetPlatformVersion>` from `8.1` to `10.0.26100.0`
- Removed `<UseOfMfc>Static</UseOfMfc>` (a command-line program does not need MFC)

## New Compilation Scripts

### `build_mic_icl.bat`

Full compilation script; pipeline: `run.c` → `bfp16.c` → `win.c` → link → `llama2_mic.exe`

### `link_mic.bat`

Link-only script, used for quick relinking without recompiling.

## Key Problems Solved

### Problem 1: MIC sub-compiler cannot find `size_t`

```
bfp16.h(71): error: *MIC* identifier "size_t" is undefined
```

**Fix:** Add `#include <stddef.h>` to `bfp16.h`

### Problem 2: MIC sub-compiler cannot find standard header files

**Fix:** Pass the Intel compiler's MIC header file directory at compile time via `/Qoffload-option,mic,compiler,"-I<MIC_INC>"`:
- `C:\...\compiler\include\mic` (stddef.h, stdint.h, etc., DOS 8.3 short paths)

### Problem 3: MIC linker has undefined MKL symbols

```
undefined reference to `sdot'
undefined reference to `cblas_sgemv'
```

**Fix:** 
- On Windows, the MIC linker tool is named `link` instead of `ld`
- Pass the MIC MKL static libraries (`.a` files) via `/Qoffload-option,mic,link,"... ... ... -lpthread"`

### Problem 4: MSVC host linker space-in-path issue

```
LINK : fatal error LNK1181: 无法打开输入文件"C:\Program.obj"
```

**Fix:** Wrap each `.lib` file path individually in quotes.

### Problem 5: cannot find `ofldbegin.obj` / `ofldend.obj`

```
ipo: error #11018: Cannot open ofldbegin.obj
```

**Fix:** Add the Intel compiler library directory `compiler\lib\intel64_win` to the `LIB` environment variable.

## Build Outputs

| File | Description |
|---|---|
| `llama2_mic.exe` (~1.5 MB) | fat binary containing the MIC offload code |
| `run_mic.obj` | MIC offload compilation output of run.c |
| `bfp16_mic.obj` | MIC offload compilation output of bfp16.c |
| `win.obj` | host compilation output of win.c |

## How to Run

```cmd
llama2_mic.exe stories110M.bin -i "Once upon a time"
```

Xeon Phi 7120P coprocessor hardware + MPSS drivers are required to actually offload execution to the MIC. Without the hardware, the code runs the MIC path on the CPU.
