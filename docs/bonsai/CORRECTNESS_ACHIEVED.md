# 正确性达成：Ternary Bonsai 2 27B 参考实现已验证

**状态：端到端精确匹配官方实现。**

```
top-1 = 11751   official = 11751 (' Paris')   MATCH
rank of 11751 in my logits = 0 / 248320   => exact match
```

参考实现（`scripts/bonsai_forward.py`）与 C 移植（`src/bonsai/model.c`）现在都逐阶段
对齐官方 `llama.cpp` fork 的输出。

---

## 两个真正的 bug

### Bug 1 —— `l2norm` 按块归一化，应为**逐头**

`model.c` 原来对 2048 维的 q 块和 k 块各算**一个**标量范数。官方的
`ggml_compute_forward_l2_norm_f32`（`ops.cpp:4195-4218`）是**每行一个 scale**，
所以 `L2_NORM(view{128,32,1,1})` = **32 个独立的 128 维头归一化**。

| 方案 | 与 oracle 的差 |
|---|---|
| 旧代码（q 一个、k 一个标量） | 2.44e-01 |
| "q+k 共享一个标量"（我最初的提议，**错误**） | 3.13e-01 |
| **32 个逐头归一化（正确）** | **0.0** |

**为什么危害大**：旧归一化让 DeltaNet 原始输出**小了 80 倍**
（`sum(o) = 0.035170` vs 官方 `2.820900`），于是 `mean(o²)` 掉到 `6.03e-09`，
**低于 `rms_eps = 1e-6`** —— 逐头 RMSNorm 于是运行在 ≈`1/√eps` = 997 而不是 484，
**归一化失效**，q/k 的尺度误差不再抵消。

修复后 `sum(o) = 2.819989` vs 官方 `2.820900`（比值 0.9997）。

### Bug 2 —— 全注意力的 Q/gate 布局（本轮）

`attn_q.weight` 输出 **12288** 维。官方图的用**逐头**布局：

```
Qcur_full-3      = MUL_MAT(blk.3.attn_q.weight{5120, 12288}, xb)  = {12288}
Qcur_full (view) = VIEW(..., {256, 24})          ← Q 部分（每头 256）
gate_reshaped-3  = CONT(Qcur_full (view){256, 24}) = {6144}
gate_sigmoid-3   = SIGMOID(gate_reshaped-3)
```

即布局是 **`[n_head, {Q(256), gate(256)}]`** —— 每头贡献 512 维，前 256 是 Q、后 256 是 gate。

我和 numpy oracle 都读成了**块布局**（前 6144 = Q、后 6144 = gate），
于是**把每个头的第二个 query 半段当成了 gate**：

| | 官方 | 块布局（错） | 逐头（对） |
|---|---|---|---|
| `gate` sum | -25014.379 | -12465.213 | **-25011.613** |
| `sigmoid(gate)` sum | **128.947** | ~1587 | **128.966** |
| `gate` first3 | `[-3.2760 -4.5583 -4.1098]` | `[0.4653 -0.0076 0.1727]` | **`[-3.2699 -4.5540 -4.1169]`** |

`sigmoid(gate)` 从 ~0.03 变成 ~0.6 —— **注意力输出被放大 3 倍**。

修复后第 3 层：

```
l_out-3 first3  我 = [0.0386  0.0267 -0.1557]   官方 = [0.0397 0.0258 -0.1553]
l_out-3 l2      我 = 34.3467                    官方 ≈ 34.370      比值 0.999
```

**逐层比值从 3.045 变成 0.999。**

---

## 最终验证结果

```
=== block 0 (GDN) 逐阶段 ===
  embed          first_err=3.39e-05   OK
  attn_norm      first_err=4.49e-05   OK
  signed         first_err=4.49e-05   OK
  qkv            first_err=7.62e-03   OK   （PQ2_0 量化固有，见下）
  conv           first_err=2.74e-04   OK

=== 第 63 层最终 hidden state ===
  first3_err = 4.60e-02   OK
      got  first3 = [8.3247 -4.3339 5.6029]
      want first3 = [8.2843 -4.3284 5.6489]
      ||x|| = 519.3618   official = 519.5392   ratio = 0.9997

=== 端到端 ===
  top-1 = 11751   official = 11751 (' Paris')   MATCH
  rank = 0 / 248320   => exact match
```

---

## 关于 `qkv` 的 0.25% 偏差

`qkv` 有约 0.25% 的相对偏差（值 ~3.0，误差 7.6e-3）。**这是该 checkpoint 的
PQ2_0 量化固有误差，不是解码 bug** —— PQ2_0 每权重 2 bit，用本实现解码或用
numpy 参考解码，残差量级相同。

**它对结果无害**：带着这个偏差，前向仍然精确复现官方的下一个 token。
所以 `test_forward.c` 里这个探针的容差按量化器定（1e-2），而不是按 dump 的
4 位小数打印精度定（2e-4）。

---

## 方法论教训（本次开发最重要的收获）

**用推理代替测量，反复导致错误结论。** 具体案例：

| 我的"推理" | 测量的真相 |
|---|---|
| "C 的状态更新有轴 bug" | **不存在** —— 转置布局下两种循环都自洽，从零初值逐位相同 |
| "那个修复是空的、新版才正确" | 也是错的 —— 是布局重参数化，两者等价 |
| "应改成 q+k 共享 L2" | **比原代码更错**（3.13e-01 vs 2.44e-01） |
| "block_attn 量级差了 258 倍" | sqrt 折算 6 个采样点的假象，真实是 ~6 倍 |
| "final_output 偏差 258 倍" | 采样点恰好落在低能量区，折算方法在我自己向量上就错 90 倍 |

**可靠的流程只有一条**：改之前先在 numpy 上把**新旧两版显式写出**并逐元素比对；
判断量级要用**全元素统计量**（dump 的 `sum = ...` 行）而不是首尾采样值。

**dump 索引的坑（共踩 5 次）**：`common_debug_print_tensor` 对**每个轴独立**取
「首 3 + 末 3」，`%12.4f` 打印。多 token 张量每 token 一行；多 head 张量按 head 分组；
4D 张量打印顺序是 `ne3 → ne2 → ne1 → ne0`。**取数前必须先写出多维索引。**

---

## 剩余工作

| 项 | 状态 |
|---|---|
| 参考实现正确性 | ✅ **完成**（本轮达成，`ALL CHECKS PASSED`） |
| 全 64 层逐层对齐 | ✅ GDN 48 层与全注意力 16 层均已对齐 |
| k1om 交叉编译 | ❌ **阻塞**：PSXE 2017 安装器是纯 GUI |
| k1om 部署与性能实测 | ❌ 依赖上一条 |
| C 版 BPE 分词器 | 未开始（目前只吃 token id 数组） |

### k1om 工具链的阻塞详情

已查清的事实：

- 安装包在 `D:\Users\Administrator\Desktop\windows\Intel_Parallel_Studio_XE_2017_win\`（12.34 GB）
  - `extracted\parallel_studio_xe_2017_setup.exe`（2.56 GiB）—— 真正的安装器
  - `extracted\license_dir\license\parallel_studio_{a..e}.lic` —— **license 齐全**
  - `vs2015.3.com_chs.iso`（7.11 GB）—— VS2015 Update 3
- 安装器**是纯 GUI**：
  - `--help` / `-?` / `-h` → 输出 0 字节，进程只挂窗口
  - `-s --output <cfg>` → 退出码 1，不生成配置模板
  - `7zr l` → "Cannot open the file as archive"（不是归档格式，无法解包提取组件）
- **LLVM 这条路走不通**：Zig 0.17 的 x86_64 CPU 列表里**没有 `k1om`**
  （只有 `knl`，而 KNL 用 AVX-512，在 Knights Corner 上会非法指令）。
  `-target ... -k1om` 报 `UnknownOperatingSystem`。
- 全盘搜索 `icc.exe` / `icl.exe` / `k1om-mpss-linux-gcc.exe` → **均未找到**。
- `docs/wsl-k1om-toolchain.md` 提到之前用过 **PSXE 2020 + `icl /Qmic`** 且
  "sysroot 已复制到 `C:\Temp\llama\`"，但**这些路径现在都不存在**，环境已丢失。

**需要的决策**：Intel 静默安装要求一份配置文件（`--silent <cfg>`），
而生成模板的开关没能试出来。可选路径：
1. 在交互式桌面会话里手动跑一次 GUI 向导，记下组件选择后再复用
2. 带 User-Agent 抓取 Intel 的静默安装指南 PDF，手工构造 cfg
3. 找一个预构建的 k1om 交叉工具链

### 已具备的上卡基础设施（之前会话留下的）

`deploy\` 下已有可用脚本：`upload_k1om.ps1`（scp `-O` 兼容 BusyBox）、
`run_k1om.ps1`（ssh 到 192.168.1.100 执行）、`bench.sh` / `bench2.sh`。
卡在线：`mic0: online`，244 核，SSH 用 `~/.ssh/id_rsa_mic` 可登录。

