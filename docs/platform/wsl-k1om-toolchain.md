# WSL k1om Cross-Compile Toolchain Usage Guide

> **2026-08-01 Update**: The WSL interop approach has been deprecated because WSL2 cannot execute Windows PE binaries (icc.exe).
> The current recommendation is **calling `icc.exe` (intel64_mic) directly from PowerShell**; the sysroot files have been copied to `C:\Temp\llama\`.
> See section 12.5 of `DEBUG_LOG.md`. WSL is now only used for SCP uploads to the 7120P.
>
> The following is the original WSL toolchain documentation.

---

## Overview

This document describes the cross-compile toolchain configured in WSL2 (Ubuntu 22.04) for the **Xeon Phi 7120P (Knights Corner, k1om)**.

The toolchain bridges Intel PSXE 2020 and MPSS 3.8.6 on the Windows side via WSL interop, making WSL the development environment for the Xeon Phi coprocessor.

## Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│  Windows Host                                       │
│                                                     │
│  ┌─── Intel PSXE 2020 ───────────────────────────┐  │
│  │   icl.exe (classic compiler, /Qmic)           │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌─── MPSS 3.8.6 ────────────────────────────────┐  │
│  │    k1om binutils (ld/as/ar/objdump, etc.)     │  │
│  │    k1om sysroot (glibc 2.21 / kernel 2.6.38)  │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌────────────────── WSL2 ────────────────────────┐  │
│  │  Ubuntu 22.04                                   │  │
│  │                                                 │  │
│  │  /opt/mpss/                                     │  │
│  │  ├── k1om-mpss-linux/         (sysroot)         │  │
│  │  └── toolchain/                                  │  │
│  │      └── bin/                                    │  │
│  │          ├── k1om-mpss-linux-gcc  → icl /Qmic   │  │
│  │          ├── k1om-mpss-linux-g++  → icl /Qmic   │  │
│  │          ├── k1om-mpss-linux-ld   → binutils    │  │
│  │          ├── k1om-mpss-linux-as   → binutils    │  │
│  │          ├── k1om-mpss-linux-ar   → binutils    │  │
│  │          ├── mic-scp              → scp→mic0    │  │
│  │          ├── mic-ssh              → ssh→mic0    │  │
│  │          └── ...                                 │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│   ┌─────────── PCIe ───────────┐                      │
│   │  Xeon Phi 7120P (mic0)     │                      │
│   │  192.168.1.100             │                      │
│   │  Linux 2.6.38.8+k1om       │                      │
│   └────────────────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Activate the Toolchain

```bash
source /mnt/c/Users/Administrator/Desktop/project/wsl-k1om-env.sh
```

Or add it to `~/.bashrc` to load it automatically:

```bash
echo 'source /mnt/c/Users/Administrator/Desktop/project/wsl-k1om-env.sh' >> ~/.bashrc
```

### 2. Compile Code

```bash
# Compile to an object file (compile only, no link)
k1om-mpss-linux-gcc -c -O3 myapp.c -o myapp.o

# Compile multiple files and link
k1om-mpss-linux-gcc -O3 file1.c file2.c -o myapp -lm

# C++ compilation
k1om-mpss-linux-g++ -std=c++11 -O3 mycpp.cpp -o mycpp
```

### 3. Upload to the Xeon Phi

```bash
# Upload the executable
mic-scp myapp /tmp/

# Upload multiple files
mic-scp myapp data.bin /tmp/

# SSH into the Xeon Phi
mic-ssh

# Execute on the Xeon Phi
mic-ssh "cd /tmp ; ./myapp"
```

## Environment Variables

| Variable | Value | Description |
|------|-----|------|
| `PATH` | Append `/opt/mpss/toolchain/bin` | Toolchain directory |
| `K1OM_SYSROOT` | `/opt/mpss/k1om-mpss-linux` | k1om target sysroot |
| `K1OM_TOOLCHAIN` | `/opt/mpss/toolchain` | Toolchain install path |
| `K1OM_CC` | `k1om-mpss-linux-gcc` | C compiler |
| `K1OM_CXX` | `k1om-mpss-linux-g++` | C++ compiler |

## Tool Reference

### Compiler

| Command | Description |
|------|------|
| `k1om-mpss-linux-gcc` | C cross-compiler (→ Windows icl /Qmic) |
| `k1om-mpss-linux-g++` | C++ cross-compiler (→ Windows icl /Qmic) |

Commonly used options supported by the compiler:

| Option | Equivalent icl option | Description |
|------|--------------|------|
| `-c` | `-c` | Compile only |
| `-O0/-O1/-O2/-O3` | `-O0/-O1/-O2/-O3` | Optimization level |
| `-g` | `/Zi` | Debug info |
| `-Wall` | `/W3` | Warnings |
| `-I<path>` | Pass through | Header file search path |
| `-L<path>` | `/link /LIBPATH:<path>` | Library search path |
| `-l<lib>` | `/link <lib>.lib` | Link library |
| `-std=c11` | `/Qstd=c11` | C standard |
| `-std=c++11` | `/Qstd=c++11` | C++ standard |
| `-fopenmp` | `/Qopenmp` | OpenMP support |
| `-shared` | `/DLL` | Shared library |
| `-Wl,<args>` | `/link <args>` | Linker arguments |

### Binutils

| Command | Description |
|------|------|
| `k1om-mpss-linux-ld` | GNU linker |
| `k1om-mpss-linux-as` | GNU assembler |
| `k1om-mpss-linux-ar` | Static library archiver |
| `k1om-mpss-linux-nm` | List symbols |
| `k1om-mpss-linux-objdump` | Disassemble object files |
| `k1om-mpss-linux-objcopy` | Convert object file format |
| `k1om-mpss-linux-readelf` | Read ELF headers |
| `k1om-mpss-linux-strip` | Strip symbols |
| `k1om-mpss-linux-size` | Section sizes |
| `k1om-mpss-linux-ranlib` | Archive index |
| `k1om-mpss-linux-strings` | Extract strings |
| `k1om-mpss-linux-addr2line` | Address→line number |
| `k1om-mpss-linux-c++filt` | Demangle C++ symbols |
| `k1om-mpss-linux-elfedit` | Edit ELF attributes |
| `k1om-mpss-linux-gprof` | Performance profiling |

### Deployment Tools

| Command | Description |
|------|------|
| `mic-scp <local> <remote>` | Upload files to mic0 (192.168.1.100) |
| `mic-ssh [cmd]` | SSH to mic0 / run remote commands |

## Directory Structure

```
/opt/mpss/
├── k1om-mpss-linux/                    # k1om sysroot (target system)
│   ├── lib/
│   │   ├── libc.so.6                   # glibc 2.21
│   │   ├── libm.so.6
│   │   ├── libpthread.so.0
│   │   ├── libgcc_s.so.1
│   │   ├── libstdc++.so.6
│   │   ├── libcoi_host.so              # COI (Coprocessor Offload Infrastructure)
│   │   └── libscif.so                  # SCIF (Symmetric Communication Interface)
│   └── usr/
│       ├── include/                    # system header files
│       │   ├── stdio.h, stdlib.h, ...
│       │   ├── scif.h
│       │   └── coi/
│       └── lib/
│
├── x86_64-mpsssdk-linux/              # host SDK (Windows exe)
│   └── usr/
│       └── bin/
│           └── k1om-mpss-linux/        # k1om binutils (.exe)
│
└── toolchain/
    └── bin/
        ├── k1om-mpss-linux-gcc         # → icl.exe /Qmic
        ├── k1om-mpss-linux-g++         # → icl.exe /Qmic
        ├── k1om-mpss-linux-ld          # → binutils
        ├── k1om-mpss-linux-ar          # → binutils
        ├── ...(other binutils)...
        ├── mic-scp                     # SCP to mic0
        └── mic-ssh                     # SSH to mic0
```

## Makefile Template

```makefile
# k1om Xeon Phi 7120P Makefile template
K1OM_CC   = k1om-mpss-linux-gcc
K1OM_CXX  = k1om-mpss-linux-g++
K1OM_LD   = k1om-mpss-linux-ld
K1OM_AR   = k1om-mpss-linux-ar
K1OM_SCP  = mic-scp

K1OM_CFLAGS   = -O3 -Wall
K1OM_CXXFLAGS = -O3 -Wall -std=c++11
K1OM_LDFLAGS  = -lm

TARGET  = myapp
SOURCES = main.c helper.c
OBJECTS = $(SOURCES:.c=.o)

MIC_TARGET = /tmp/$(TARGET)
MIC_IP     = 192.168.1.100

.PHONY: all clean deploy run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(K1OM_CC) $(K1OM_CFLAGS) $^ -o $@ $(K1OM_LDFLAGS)

%.o: %.c
	$(K1OM_CC) $(K1OM_CFLAGS) -c $< -o $@

deploy: $(TARGET)
	$(K1OM_SCP) $(TARGET) $(MIC_TARGET)

run: deploy
	mic-ssh "cd /tmp ; ./$(TARGET)"

clean:
	rm -f $(OBJECTS) $(TARGET)
```

## Building an Existing Project (llama2.knc.c)

The llama2.knc.c project can be compiled directly with MSVC + Intel PSXE on the Windows side:

```bash
# Run in WSL (invokes the Windows build script)
cd /mnt/c/Users/Administrator/Desktop/llama.c/llama2.knc.c/llama2.knc.c-master/llama2.knc.c-master
cmd.exe /c build_msvc.bat
```

Or compile it manually with the WSL toolchain:

```bash
source /mnt/c/Users/Administrator/Desktop/project/wsl-k1om-env.sh

k1om-mpss-linux-gcc -O3 -Wall \
    -I/mnt/c/Users/Administrator/Desktop/llama.c/llama2.knc.c/llama2.knc.c-master/llama2.knc.c-master \
    *.c -o llama2.knc -lm -fopenmp
```

## FAQ

### Q: The compiler reports "Exec format error"

Make sure WSL interop is enabled:

```bash
cat /proc/sys/fs/binfmt_misc/WSLInterop
# Expected output: enabled
```

If it is not enabled, check `/etc/wsl.conf`:

```ini
[interop]
enabled = true
appendWindowsPath = true
```

Then exit all WSL sessions and run the following in PowerShell:

```powershell
wsl --shutdown
```

### Q: icl.exe fails to compile: header files not found

`icl.exe` requires the MSVC environment. Verify MSVC is available in Windows PowerShell:

```powershell
# Load the VS 2022 environment
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
icl
```

### Q: mic-scp connection refused

Make sure the Xeon Phi coprocessor is powered on and the IP is correct:

```bash
# Default IP: 192.168.1.100
# You can check its status in Windows with micctrl
micctrl -s
```

### Q: How do I check whether an object file is k1om architecture?

```bash
k1om-mpss-linux-objdump -f myapp.o | grep architecture
# Expected output: architecture: k1om
```

### Q: How do I share SSH keys between WSL and Windows?

`mic-scp` and `mic-ssh` automatically copy the key from `C:\Users\Administrator\.ssh\id_rsa_mic` into WSL. If you do not have a key yet, generate one first:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_mic -N ""
# Copy the public key to the Xeon Phi
mic-scp ~/.ssh/id_rsa_mic.pub /tmp/
mic-ssh "cat /tmp/id_rsa_mic.pub >> /root/.ssh/authorized_keys"
```

## Version Information

| Component | Version |
|------|------|
| WSL2 | 2.7.11.0 |
| Ubuntu | 22.04.5 LTS |
| MPSS | 3.8.6 |
| Intel PSXE | 2020.4.311 |
| k1om binutils | 2.22-7 |
| k1om glibc | 2.21 |
| k1om kernel | 2.6.38.8+gfa9c0d9 |
| Xeon Phi | 7120P (Knights Corner, KNC) |
