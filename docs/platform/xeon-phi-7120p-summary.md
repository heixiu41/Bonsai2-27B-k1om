# Intel Xeon Phi 7120P Technical Summary

> Generated: 2026-07-30

---

## 1. Hardware Overview

| Item | Details |
|------|------|
| Model | Intel Xeon Phi 7120P (C0PRQ-7120 P/A/X/D) |
| Serial number | ADKC51100245 |
| Cores | 61 |
| Frequency | ~1.24 GHz (1238095 kHz) |
| Stepping | C0 (Stepping ID: 2) |
| Memory | 15872 MB GDDR5 (Samsung, 4096 Mb density, GDDR5 @ 5.5 GT/s) |
| ECC | Enabled |
| TDP | 300W passive cooling |
| Voltage (Core) | 1024000 uV |
| Voltage (GDDR) | 1501000 uV |
| Current temperature | ~51°C |

## 2. PCIe Connection

| Item | Details |
|------|------|
| Vendor ID | 0x8086 (Intel) |
| Device ID | 0x225c |
| Subsystem ID | 0x7d95 |
| PCIe lanes | x16 |
| PCIe speed | 5 GT/s |
| Max Payload | 256 bytes |
| Max Read Request | 512 bytes |

## 3. Firmware Versions

| Component | Version |
|------|------|
| Flash Version | 2.1.02.0391 |
| SMC Firmware | 1.17.6900 |
| SMC Boot Loader | 1.8.4326 |

## 4. Driver and Software Stack

| Item | Details |
|------|------|
| Host OS | Windows 10 Pro for Workstations |
| MPSS version | 3.8.6.6348 |
| Driver version | 3.8.6.6348 |
| Coprocessor OS | Linux 2.6.38.8+mpss3.8.6 (k1om architecture, built 2019-07-25) |
| Distribution | Intel MIC Platform Software Stack (Built by Poky 7.0) 3.8.6 |

## 5. Network Configuration

| Item | Details |
|------|------|
| Virtual NIC | Intel(R) Xeon Phi(TM) coprocessor Virtual Ethernet Adapter (x64) |
| Host IP | 192.168.1.99 / 255.255.255.0 |
| Coprocessor IP | 192.168.1.100 |
| No DHCP | Static configuration |

## 6. Coprocessor Linux Environment

| Item | Details |
|------|------|
| Hostname | mic0 |
| Total memory | 15514 MB |
| Free memory | ~15222 MB |
| Total disk | 12.9 GB |
| Disk used | 62.5 MB |
| Swap | None |
| GCC | Not installed |
| Intel development tools (/opt/intel) | Not installed |

## 7. SSH Connection Methods

### Method 1: OpenSSH (recommended)

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_rsa_mic" root@192.168.1.100
```

### Method 2: PuTTY

PuTTY path: `C:\Program Files\PuTTY\putty.exe` (v0.84)

First convert the `id_rsa_mic` private key to `.ppk` format with PuTTYgen, then import it.

### SSH Key Location

| File | Path |
|------|------|
| Private key | `%USERPROFILE%\.ssh\id_rsa_mic` |
| Public key | `%USERPROFILE%\.ssh\id_rsa_mic.pub` |
| Algorithm | RSA 4096-bit |
| Comment | mic-access-20260730 |

## 8. MPSS Management Commands

| Command | Purpose |
|------|------|
| `micctrl -s` | View coprocessor status |
| `micctrl -r mic0` | Reset the coprocessor |
| `micctrl -b mic0` | Boot the coprocessor |
| `micctrl -w mic0` | Wait for the coprocessor to become ready |
| `micctrl -l mic0` | View the coprocessor boot log |
| `micctrl --shutdown mic0` | Shut down the coprocessor |
| `micctrl --addssh <user> -f <keyfile>` | Add an SSH public key |
| `micinfo` | View coprocessor hardware information |
| `miccheck` | Run a diagnostic check |

## 9. Development Notes

- The coprocessor uses the **k1om** architecture (not x86), so it must be cross-compiled with the Intel Compiler
- Cross-compile flags: `icc -mmic` or `icpc -mmic`
- After compiling, the binary can be uploaded to the coprocessor via SCP for execution
- The coprocessor currently runs a minimal Linux environment with no compiler and no extra tools; it is recommended to compile on the host side

## 10. Reliable Upload Method (2026-08-01)

The `scp` in newer Windows OpenSSH uses the **SFTP subsystem** by default, but KNC's BusyBox does not provide SFTP, causing uploads to fail:

```
scp -i ~/.ssh/id_rsa_mic xxx root@192.168.1.100:/tmp/
# Connection closed  (exit 255)
```

Fix: explicitly specify the **legacy scp protocol**:

```
scp -O -i ~/.ssh/id_rsa_mic xxx root@192.168.1.100:/tmp/
```

Notes:
- After uploading an ELF, run `ssh ... "chmod +x /tmp/xxx"` (scp does not preserve the executable bit by default)
- If `REMOTE HOST IDENTIFICATION HAS CHANGED` is reported: the host key has changed (e.g., the Phi's system was reflashed); remove the old known_hosts entry and reconnect

## 11. Authoritative Verification of KNC Intrinsics Availability (2026-08-01)

> Source: PSXE 2017 compiler headers `immintrin.h`/`zmmintrin.h` + probe measurements (two-stage verification: compile/link).

### Available (passes both compile and link)

| intrinsic | Description |
|---|---|
| `_mm512_load_ps` / `_mm512_store_ps` | 64B-aligned load/store (`vmovaps`) |
| `_mm512_setzero_ps` / `_mm512_set1_ps` | Zeroing / broadcast |
| `_mm512_add_ps` / `_mm512_fmadd_ps` | Arithmetic; FMA supported in hardware |
| `_mm512_exp_ps` | **SVML exponential (KNC has hardware `vexp2ps`) → softmax can be vectorized** |
| `_mm512_load_epi32` / `_mm512_reduce_add_epi32` | Integer 512-bit load / horizontal reduction |
| `_mm512_castps_si512` | Bit conversion |
| `_mm512_mask_mov_ps` | Masked move |
| `_mm512_swizzle_ps` | KNC-specific swizzle (enumerated as `_MM_SWIZ_REG_*`) |
| `_mm512_reduce_add_ps` | Horizontal reduction |

### Compiles but fails to link (undefined reference)

| intrinsic | Actual symbol |
|---|---|
| `_mm512_loadu_ps` / `_mm512_storeu_ps` | KNC has no unaligned load variant; you must ensure 64B alignment yourself |
| `_mm512_cvtepi32_ps` | Expands to `_mm512_cvt_roundepi32_ps`, which does not link → int8 dequantization must be scalar |

### Unavailable (implicit declaration)

- `_mm512_maddubs_epi16` (required for the Q8_0 dot product; unavailable → use scalar dequantization)
- `_mm512_castsi512_si128` (requires `__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES`)
- `_mm512_exp2a23_ps`, `_mm512_rsqrt14_ps`, `_mm512_rsqrt28_ps`

### Important Limitations

- KNC **rejects SSE2 128-bit instructions** (`movq`/`movdqa` → error #13393): if the intrinsics generate 128-bit SSE, compilation fails; you must stay on the 512-bit path throughout
- All `_mm512_*` declarations are in `zmmintrin.h` (included at the end of `immintrin.h`)

## 12. Performance Optimization Results Summary (2026-08-01)

| Optimization | Result |
|---|---|
| Q8_0 matmul vectorization | +55% |
| Q8_0 matmul with 4 independent accumulators | Low thread counts: +13~16% |
| FP32 `matmul_imci` with 4 independent FMA accumulators | Low thread counts: +33~56% |
| Attention score/value multi-accumulator + FMA | Low thread counts: +7~13% |
| softmax vectorization (`_mm512_exp_ps`) | 244 threads: +2.9% (no gain at low thread counts; kept) |
| RoPE lookup table | No gain; reverted |
| `-O3` / `-fp-model fast=2` | No gain; reverted (kept the `-Fast` switch) |

Optimization rationale: multiple accumulators break the serial FMA dependency chain to improve ILP; the gains are significant at low thread counts, while high thread counts (122/244) approach a plateau.
