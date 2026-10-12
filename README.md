# Bonsai 2 27B on Intel Xeon Phi 7120P (k1om)

Ternary Bonsai 2 27B（PrismML 的三值压缩模型，基于 Qwen3.8-27B）在 Intel Xeon Phi 7120P
（Knights Corner / k1om）协处理器上的**原生 C 推理引擎**。

**状态：模型可在此卡上原生运行，输出与官方实现精确一致。支持交互式对话。**

```
top-1 = 11751   official = 11751 (' Paris')   MATCH
rank of 11751 = 0 / 248320
性能: 3.73 tok/s decode (默认 244 线程)
对话: python scripts\chat.py   ← 纯 stdlib 分词器 + 卡上常驻服务，SCIF 传模型
```

> ## 🚨 性能工作前必读
>
> **先做端到端 profile。** 曾经有 8 轮工作全部优化解码器，而它当时只占 **11.6%** ——
> 真正的瓶颈是 GDN 的 DeltaNet 递推（**70.2%**），而且它是**纯串行**的
> （线程扫描显示耗时从 1 到 244 线程完全平坦）。
> **四处 `#pragma omp` 就拿到了 5.64x**，没有动解码器一行。
>
> **但"解码器不重要"这个结论随后也过期了。** 串行瓶颈消除后，
> 解码器重新变成最大的单项（**39%**），于是 V12（40 → 30 ops/128 权重）才变得有意义。
> **两条都做了。**
>
> 教训不是"别优化解码器"，而是：**先测出瓶颈在哪，再决定优化什么** ——
> 各部分的份额会随着其他部分的改善而改变。

## 性能历程（全部卡上实测）

```
0.292 tok/s  原始
0.962        + DeltaNet 递推 head-parallel      （四处 pragma 之一）
1.257        + Hadamard 块并行
1.407        + conv 通道 + 逐头 L2 norm
1.644        + 默认线程 122 -> 244（串行消除后 244 才成为最优）
1.711        + V12 解码器（缩放折进 LUT + 两张 LUT + 输入置换）
2.686        + 小权重加载时缓存（~85 ms/token 的重复反量化消除）
             + 同一输入的 Hadamard 去重（401 → 258 次/token）
             + GDN 剩余串行 glue 并行化 + DeltaNet 状态布局转置（消 vgatherdps）
             + fwht 蝶形与符号/归一化循环 IMCI 向量化
3.198        + F3 解码器（B2 gather 取 scale + 单累加器块循环；同二进制 A/B
             内核 1.35x @244 线程，33024+ 项检查 0 失配）
3.499        + 双子智能体源码审计（A: model.c / B: pq2_0.h）落地项：
             xp 置换每次 staging 构建一次（401 → ~258 次/token，~20 ms 串行消除）
             GDN 六段/attention 五段各并成一个 omp region（352 → ~130 glue
             region/token，~8 ms）；attention 24 头并行（消唯一的 O(ctx) 串行段）
             rmsnorm 卡上双精度 8 路向量归约；rope 频率表缓存；
             argmax 向量化；OMP 预热挪进加载期（token 0: 0.72 → 0.33 s）
3.73         + drow 行缩放懒预计算：每张量首次 mm_run 时建好全行 fp16 缩放表
             （内核循环不再逐行 gather+展开；探针同二进制 A/B 1.142x @244t，
             逐位一致；+799 MB 常驻；懒建表避开加载内存峰值 OOM）
             = 12.8x
```

附：prefill 稳态 = decode（268 ms/token）；token 0 的一次性 OMP 预热
（~350 ms）已挪进模型加载期（预热后 token 0 = 0.33 s）。

**验收**：宿主回归 `ALL CHECKS PASSED`（5 个阶段检查 + 4 个隐藏状态地标，exit 0），
卡上 `top-1 = 11751 MATCH`、`rank 0/248320`。

---

## 先读这个

**→ [`HANDOVER.md`](HANDOVER.md)** —— 交接文件。包含：
- 立即可用的构建与部署配方
- 已确认正确的部分（不要破坏）
- 分级缺陷清单
- **我犯过的错**（避免重复）

然后按需读 `docs/`。

---

## 目录结构

```
src/                     核心（唯一需要编译的部分）
  pq2_0.h                PQ2_0 编解码 + matvec 内核 + Hadamard
  model.c / model.h      前向传播
  gguf.c / gguf.h        GGUF 容器加载（7.2 GB，64 位偏移）
  main_k1om.c            k1om 入口
  test_forward.c         宿主正确性门槛（4 个隐藏状态地标 + 5 个阶段检查）
  test_gguf.c            加载器自测（41 项检查）
  k1om_e2e.c             卡侧端到端验收
  probes/                67 个独立探针/基准（不参与正式构建）
      pq2cost.c          解码器成本阶梯
      pq2isa2.c          指令吞吐与语义实测
      pq2f3.c            最快解码器候选（33024 项检查 0 失配）
      pq2prim*.c         机器模型（每条 512 位指令成本相同）
      pq2omp.c           OpenMP parallel region 开销
      pq2settle.c        正确的链接探测写法（结果可观察 + 数值核对）
      pq2map/pq2sem/...  语义实测探针
      mkrun.ps1          编译+上传+运行单个探针

docs/
  bonsai/                17 份 —— 本项目本身（架构、成本分析、审查、验收）
  platform/              7 份 —— KNC 平台知识（工具链、传输通道、SSH、WSL）
  archive/               3 份 —— 早期 Qwen 工作（已被 Bonsai 取代）

scripts/                 Python 参考实现 + 卡上原始证据
  bonsai_ref.py          GGUF 解析 + selftest
  bonsai_forward.py      numpy oracle（RoPE 已修正为前后半配对）
  bonsai_diff_gt.py      与官方 dump 逐张量对比
  *_card_evidence.txt    卡上原始输出（V9 / V12 / ISA）

deploy/                  卡上部署脚本
tools/                   mic.ps1（上传+运行）、probe_links.ps1（链接探测）

ref/                     参考源码（从 llama.cpp fork 提取的语义依据）
  qwen35.cpp             Qwen3.5 架构实现
  delta-net-base.cpp     GDN（Gated DeltaNet）实现
  forkroot.txt           fork 路径记录
audit/                   畸形 GGUF 测试样本 + 审查探针
```

---

## 构建

### 交叉编译到 k1om

```powershell
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'

Push-Location src
& $icc -mmic -O2 -std=c11 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE `
       main_k1om.c model.c gguf.c -o bonsai_k1om -lm
Pop-Location
```

产出应为 ELF `machine = 0x00B5`（`EM_K1OM`）。**不是** 0x3E。

### 宿主机（正确性门槛）

```powershell
$zig = (Get-ChildItem D:\zig -Recurse -Filter 'zig.exe' | Select-Object -First 1).FullName
Push-Location src
& $zig cc -O2 -std=c11 -o tf.exe test_forward.c model.c gguf.c
Pop-Location
# 约 19 分钟；期望 ALL CHECKS PASSED + top-1 11751 + l_out-63 OK
.\tf.exe D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf
```

### 部署到卡

见 [`HANDOVER.md`](HANDOVER.md) §3。要点：
- `scp -O -c chacha20-poly1305@openssh.com`（BusyBox 无 sftp 子系统、KNC 无 AES-NI）
- `chmod +x` **必须**
- 运行需要 `LD_LIBRARY_PATH=/tmp`（OpenMP 运行库在那）
- 也可用 `tools\mic.ps1` 一键上传+运行

---

## 模型与依赖

```
模型    D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf
        7,206,168,928 字节   md5 e119192844abba9e7821f6b0e23022a3
ground truth  D:\bonsai\eval_callback_full.txt（5-token dump）
官方参考      D:\llama-prism-win\（46 个官方二进制）
fork 源码     D:\llama-prism-src\llama.cpp-prism\

硬件    Xeon Phi 7120P：61 核 / 244 线程 @ 1.238 GHz，15.5 GB
        MPSS 3.8.6，Linux 2.6.38.8，BusyBox，IP 192.168.1.100
工具链  PSXE 2017 (ICC 17.0.0) + MPSS k1om sysroot
        Zig 0.17.0（提供宿主机 clang；宿主无 MSVC/gcc）
```

---

## 关键知识（血泪教训）

| 知识 | 说明 |
|---|---|
| **top-1 不是正确性判据** | 实测 RoPE 配对错误时 top-1 仍然 MATCH。必须看 `l_out-63` 隐藏状态 |
| **每条 512 位指令成本相同** | KNC 上 `vpaddd == vpermd == vfmadd`。唯一货币是**指令数** |
| **头文件声明 ≠ 能链接** | `cvtepi32_ps`/`shuffle_epi8`/`madd_epi16`/字节字操作全部不存在 |
| **链接探测有陷阱** | 结果不被使用 ⇒ 死代码消除 ⇒ 假阳性。必须让结果可观察 |
| **编译器 CSE 会让时基无效** | 多个迭代算出相同结果 ⇒ 被合并。"1.37x" 就是这么来的假象 |
| **k1om 是 LP64** | `sizeof(long)==8`，`ftello` 处理 3 GB 无问题 |
| **访存不是瓶颈** | 实测 196 GB/s，每 token 只需 36 毫秒（占 1%） |
| **内核没有神秘的"模型内减速"** | 所谓 4.3x 差距是 GB/s 误标成 G 权重/s（换算系数恰为 3.76x）。真实差距 1.15x，由 shape 混合解释（pq2call 实测） |

---

## 文档索引

### 项目本身（`docs/bonsai/`）

| 文档 | 内容 |
|---|---|
| `BONSAI2_FORWARD_SPEC.md` | **前向语义权威说明**（带 fork 源码行号） |
| `BONSAI2_K1OM_PLAN.md` | 早期计划（多处已过时，有更正头） |
| `CORRECTNESS_ACHIEVED.md` | 端到端正确性达成过程与两个真 bug |
| `CODE_AUDIT_RESULTS.md` | 三轮独立代码审查汇总 |
| `PQ2_DECODER_COST_ANALYSIS.md` | 解码器成本分析与优化探索（含负面结论） |
| `PQ2_V12_RESULT.md` | V12 解码器（1.55-1.58x，验证为精确） |
| `pq2_0_imci_decoder.md` | 向量化解码器完整报告 |
| `WORKORDER_PARALLEL_OVERHEAD.md` | **下一步作业说明**（parallel region 开销） |
| `FINAL_STATUS_K1OM.md` | 交付状态（含更正头） |
| `ENGINE_TWEAKS_AB.md` | README 三条剩余杠杆的实测结论（含一条负结果与一条默认关闭） |
| `REPACK_CODES_ONLY.md` | codes-only 重打包（34→32 字节/块）：lane 映射证明、内核 A/B、端到端无收益 |
| `PERFORMANCE_OPTIMIZATION.md` | 性能分析（已修正带宽错误） |
| `GDN_ROOTCAUSE.md` / `GDN_AXIS_AND_L2_FINDINGS.md` / `_AUDIT_GDN_C_vs_PY.md` | GDN 递推的调试过程 |
| `KNC_INTEGER_SIMD.md` | KNC 指令可用性实测清单 |
| `K1OM_BUILD_RECIPE.md` | 交叉编译配方 |
| `COI_ROUTE.md` | COI 路线（宿主侧 ctypes 可驱动） |

### 平台知识（`docs/platform/`）
工具链搭建、传输通道实测、SSH 主机密钥、WSL 环境、卡的整体摘要。

### 归档（`docs/archive/`）
早期 Qwen 相关工作，已被 Bonsai 取代。

---

## 已知缺口

| 项 | 状态 |
|---|---|
| **性能** | 3.73 tok/s decode。matvec 内核 ~63%（已处于 shape 混合后的满速），drow 懒预计算已落地（+799 MB 常驻，逐位一致）。README 上剩的三条杠杆已逐条做完，见 `ENGINE_TWEAKS_AB.md`：**prefill 输出头跳过已落地（+5.3% prefill，32 token 贪心序列逐 token 一致）**；**块内 rescale 变体已实现但默认关闭**（同二进制 A/B 两次跑出 +2.72% / −2.10%，落在 ±6% 跨进程噪声内，收益未确证，`BONSAI_RESCALE=1` 可开）；**vexp223ps 向量化 exp 是负结果**——icc 早就把逐元素 `expf` 变成 `__svml_expf16`，手写实测 1.03x，别再查。另做了 **codes-only 重打包**（34→32 字节/块，见 `REPACK_CODES_ONLY.md`）：机制**已证明且逐位一致**、内核**单线程 +7.8%…+9.2%**，但 244 线程端到端**效应落在噪声内**（交错的同二进制 A/B：3.715 → 3.739 tok/s，+0.66%，噪声地板约 ±0.6%），故 `BONSAI_PACK=1` 可选、默认关闭；"省 0.4 GB 常驻"**不成立**（空洞散布在同一分配里，RSS 仍是 9.03 GiB）。附带真实改进：每块越界读 48 字节的隐患消失 |
| **C 版分词器** | ✅ 已有 Python 纯 stdlib 版（`scripts/chat.py`，含 chat template 与流式解码）；C 版仍未做 |
| **COI 缓冲映射** | 未做 —— 完成可省掉模型上传并腾出 6.8 GB 卡内存 |
| **长上下文** | `pos > 5` / `ctx > 256` 未测试 |
| **多 token 生成** | 只验证过 5 prefill + 3-4 decode |

---

## 免责说明

这个项目的结论来自**实测**，但过程中也产生过错误结论。所有已更正的错误都在对应文档顶部
标了更正头，`HANDOVER.md` 里有完整的"我犯过的错"清单。

**看到没有更正头的性能数字时，先确认它是端到端实测还是合成基准。**
这个项目曾经因为合成基准的参数不变被编译器合并，得出过一个虚假的 101.7 G weights/s。
