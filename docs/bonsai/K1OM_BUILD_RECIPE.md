# k1om 交叉编译 —— 可用配方（已验证）

> ## ⚠️ 两处更正（本文件早期版本有错）
>
> **1. k1om 是 LP64，`long` 是 64 位。** 早期版本把"32 位 `long`"列为关键坑，**这是错的**。
> 实测：在 mic0 上 `sizeof(long)=8`、`sizeof(off_t)=8`、`ftello` 能处理 3 GB 文件并返回
> 3145728000；**去掉** `-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE` 行为完全一致。
> k1om sysroot 的 `bits/wordsize.h` 写着 `__WORDSIZE 64`。
> 那两个宏仍然保留在下面的命令里 —— 它们**无害**，只是在这个目标上不必要。
> （32 位截断的危险只存在于 i386-linux-gnu 之类的 ILP32 目标。）
>
> **2. `[threads]` 参数现在真的会被解析。** 早期版本的文档提到它但代码从未读取；
> 现在按 `显式参数 > OMP_NUM_THREADS > 默认 122` 的优先级生效，非法值 → usage + exit 2。
> 另外 mode 现在是**白名单**（只接受 `selftest` / `forward`），不再是"非 selftest 即 forward"。

**状态：`bonsai_k1om` 已在 Xeon Phi 7120P 上原生运行。**

```
$ /tmp/bonsai_k1om
usage: /tmp/bonsai_k1om <model.gguf> [selftest|forward] [threads]
```

---

## 工具链位置（PSXE 2017 装在 C 盘）

| 组件 | 路径 |
|---|---|
| 编译器 | `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe` |
| 链接器 | 同目录 `xild.exe` |
| 环境定义 | 同目录 `x86_64-linux.env` |
| **sysroot** | **`C:\Program Files\Intel\MPSS\k1om-mpss-linux\`** ← MPSS 自带 |
| dyn sysroot | `C:\Program Files\Intel\MPSS\x86_64-mpsssdk-linux\` |
| OpenMP 运行库 | `...\windows\compiler\lib\intel64_win_mic\libiomp5.so`（k1om ELF，1.3 MB） |
| 执行工具 | `C:\Program Files\Intel\MPSS\bin\micnativeloadex.exe` |

编译器版本：`icc (ICC) 17.0.0 20160721`

## 编译与链接

```powershell
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'

# 必须的宏：没有它们 ftello/fseeko 会隐式声明 -> 32 位截断 -> 7.2 GB 文件读不了
$flags = '-mmic -O2 -std=c99 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE'

& $icc $flags.Split(' ') main_k1om.c model.c gguf.c -o bonsai_k1om -lm
```

产出验证：ELF `machine = 0x00B5`（181 = `EM_K1OM`）。**不是** 0x3E（x86-64）。

## 部署与运行

```powershell
$key = "$env:USERPROFILE\.ssh\id_rsa_mic"
# BusyBox 没有 SFTP，scp 必须带 -O（legacy 协议）
scp -O -i $key bonsai_k1om    root@192.168.1.100:/tmp/
scp -O -i $key libiomp5.so    root@192.168.1.100:/tmp/

ssh -i $key root@192.168.1.100 "export LD_LIBRARY_PATH=/tmp; /tmp/bonsai_k1om /tmp/model.gguf forward"
```

## 踩过的坑

1. **`ftello`/`fseeko` 隐式声明** —— 不加 `-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE`
   会静默按 32 位处理，对 7.2 GB 文件致命。必须加。
2. **缺 `libiomp5.so`** —— 链接时找到的是 **Windows 版**（`intel64_win\libiomp5md.lib`），
   卡上跑不了。必须把 `intel64_win_mic\libiomp5.so` 一起传上去。
   `-static` 不可行：k1om 的 `libiomp5.a` 不存在（只有 `intel64_lin` 的 Linux x86-64 版）。
3. **`_POSIX_C_SOURCE`** —— 用 `clock_gettime`/`struct timespec` 需要在所有 `#include`
   之前定义 `#define _POSIX_C_SOURCE 200809L`。
4. **BusyBox 的 `head`** 不支持 `-6`，用 `head -n 6`。
5. **`bonsai_model_open(path, ctx)`** 第二个参数是 `int32_t` 上下文长度，**不是 void***。

## 卡的状态

```
Linux mic0 2.6.38.8+mpss3.8.6 #1 SMP ... k1om GNU/Linux
nproc = 244
mic0: online
```

## 下一步

1. 把 7.2 GB 模型传到卡上（`/tmp` 在重启后清空，持久化需 `mic0.filelist`）
2. 跑 `selftest` 验证 64 位文件偏移在 k1om 上正确
3. 跑 `forward` 验证 IMCI 内核（`pq2_0.h` 的 `_mm512_*`，**这是它们第一次在真硬件上执行**）
   并测 tok/s（预期 1–4）
