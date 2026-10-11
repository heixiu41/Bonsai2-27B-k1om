# 交接文件 —— k1om Bonsai 2 27B 推理引擎

> ## 🚨 先看这一条：**不要再优化解码器**
>
> **实测（`docs/K1OM_TIME_BUDGET.md`，带插桩的 profile）：**
> ```
> GDN 块 (48/token) 25.897 s  82.5%
>   DeltaNet 递推 (串行) 22.031 s  70.2%   ← 真正的瓶颈
> FFN 块             4.366 s  13.9%
> 全注意力块          0.666 s   2.1%
> mm_folded 总计     6.644 s  21.2%
>   └ PQ2_0 解码器    3.651 s  11.6%   ← 只有 11.6%
> ```
>
> **解码器只占端到端的 11.6%。** 前面 8 轮（V3/V7/V8/V9/V10/V12/F3）
> 全部在优化它，若全部集成，端到端收益约 **+1.4%**。
>
> **决定性证据是线程扫描** —— DeltaNet 递推的耗时**从 1 到 244 线程完全平坦**
> （11.891 / 11.883 / 11.846 / 12.230 / 13.564 s），即纯串行；
> 而 matvec 从 170.4 s 降到 1.74 s。**平坦的曲线就是答案。**
>
> **已修复并实测**（4 处 `#pragma omp`，见 §0b）：**0.292 → 1.646 tok/s = 5.64x**，
> 且没有动解码器一行。
>
> **做任何性能工作之前，先做端到端 profile。** 我从第 2 轮起就*假设*了
> "端到端 = 解码"而从未验证 —— 那是这个项目最大的一次方向性错误。

---

## 0b. 性能优化：串行 glue 并行化（5.64x，已验证）

`docs/K1OM_TIME_BUDGET.md` 是权威报告。四处改动，全部在卡上实测：

| # | 位置 | 改动 | decode tok/s | 累计 |
|---|---|---|---|---|
| 0 | 基线 | — | 0.292 | 1.00x |
| 1 | `model.c` `block_gdn` DeltaNet 递推 | `#pragma omp parallel for` over 48 个 value head | 0.962 | 3.29x |
| 2 | `pq2_0.h` `hadamard_forward` | 块循环并行（块内蝶形保持串行） | 1.257 | 4.31x |
| 3 | `model.c` conv 通道循环 + 逐头 L2 norm | 各自加 pragma | 1.407 | 4.82x |
| 4 | `main_k1om.c` `DEFAULT_THREADS` | 122 → **244**（串行去掉后新最优） | **1.646** | **5.64x** |

**线程扫描（新版本）**：
```
 61 线程  1.009 tok/s
122 线程  1.396
183 线程  1.490
244 线程  1.634   ← 新的默认值
（旧版 244 比 122 慢，因为那时被串行部分主导）
```

**为什么递推慢**（证据，非推测）：`Sh[a*st+i]` 走 128 行、每行相隔 512 字节，
ICC 把它变成 **64 个 `vgatherdps` 重试循环**（标注 `Latency 45`）——
KNC 上最坏的指令类。每个 FMA 约 21.6 ps（约 15 周期）。

**正确性**：全程 `top-1 = 11751 MATCH`、`rank 0`；宿主回归已重跑。

**剩余机会**（profile 实测，未做）：
```
+11%  PR_GDN_BETA 里 4 次 dequant_to_f32（每层 2.6M 元素 × 48 层 = 22528 个
      小权重被重复反量化 48 次）—— 未拆解，可能是下一个目标
天花板：不可约串行工作 ≈2.94 s/token
```

---

## 0c. 第 N 轮：小权重缓存 + Hadamard 去重 + glue 收尾（1.711 → 2.686 tok/s，已验收）

在 V12 集成（1.711 tok/s）之后做的。全部改动在 `src/model.c` / `src/model.h` /
`src/pq2_0.h`，宿主回归 `ALL CHECKS PASSED`（`l_out-63` first_err=4.60e-02 与改动前
**逐位相同**），卡上 `top-1 = 11751 MATCH`、`rank 0`。

| 改动 | 内容 | 数值性质 |
|---|---|---|
| ① 小权重缓存 | 每 token 重复反量化的 11 类常量权重（beta/alpha 各 5120×48 BF16/层、conv1d、ssm_a/dt/norm、各 RMSNorm 权重）改为 `bonsai_wcache` 在 open 时经**同一条** `dequant_to_f32` 缓存一次（~110 MB）；beta/alpha 两个串行标量 matvec 融合为一次 96 行并行 matvec | 位一致（同一函数产出） |
| ② Hadamard 去重 | `mm_folded` 拆成 `mm_stage` + `mm_run`；qkv+gate、q/k/v、gate+up 共享同一次 staging → 401 → 258 次/token | 位一致（同算少算几遍） |
| ③ glue 收尾 | 逐头 gated RMSNorm、silu(z)·o 置换、FFN silu 三处加 pragma；**DeltaNet 状态转置** `Sh[a*st+i]→Sh[i*st+a]`（消 vgatherdps，纯内部布局约定） | 宿主位一致；卡上归约向量化可能 ~1e-7 漂移，实测无影响 |
| ④ fwht 向量化 | `pq2_0.h`：蝶形 h≥16 的 6 级 + 符号/归一化循环 `_mm512` 化；`(block&15)==0 && 64B 对齐` 守卫，否则回退标量 | 位一致（lane 内同样的单次加/减/乘） |

**卡上结果**：decode **2.686 tok/s**（372 ms/token），prefill 稳态相同。

**本轮两个实测结论（都更正了旧认知）**：

1. **"matvec 内核模型内比独立基准慢 4.3x" 是单位错误，不存在。** 旧数字
   "6.80 GB / 0.263 s = 25.9 G w/s" 把 **GB/s（打包字节）当成了 G 权重/s**；
   真实值 25.9 × 128/34 = **97.5 G w/s**，与独立基准真实 shape 混合下的
   99.06 G w/s（pq2call）一致。内核一直在满速跑；region 开销、工作集、
   调用开销全都被各自的探针正确排除了。更正注记已加进 `pq2reg.c` 头部。
2. **prefill 稳态 = decode。** 短 prompt 测试中 prefill 偏慢是 token 0 的
   一次性 ~350 ms warmup（OpenMP 启动等）摊进了 5 个 token 的平均值。
   逐 token 计时已加进 `main_k1om.c`。

**当前瓶颈构成**（372 ms/token → F3 后 313 ms/token）：matvec 内核（shape 混合后
的满速），其余为 glue。**F3 已集成**：同二进制 A/B 内核 1.35x @244 线程
（V12 103.9 → F3 140.4 G w/s，顺序无关），33024+ 项精确性检查 0 失配，卡上
decode **2.686 → 3.198 tok/s（+19.1%）**，top-1 MATCH、rank 0。证据与集成说明见
`src/pq2_0.h` 的 F3 注释块和 `src/probes/pq2ab.c`。解码器指令数这条线到此
基本采尽；剩余杠杆：prefill 非末位 token 跳过 output head、rope 频率表缓存。
**批量 prefill 已实测封死**（`src/probes/pq2f3batch.c`，F3 内核 + 244 线程 +
逐位一致对照：B=2/4/8/16 加速比 0.91/0.99/0.97/0.83x）——满占用时解码指令被
完全掩盖，每 token 的 8 次 xp 载入 + 8 FMA + 1 add 不可摊薄。**prefill 上限 =
decode 单 token 成本。**

**已知不兼容**：`prof/src/model.c` 是旧快照，引用被移除的 scratch 字段，
不再能对新 `model.h` 编译；需要 profile 时得同步它。

---

## 0d. 交互式对话 + SCIF 传输（已验收）

**对话链路**：`scripts/chat.py`（宿主，纯 stdlib：GGUF 里读 tokenizer、
GPT-2 字节级 BPE、Qwen pre-split、ChatML 拼装、流式增量 UTF-8 解码）→ ssh 长连接
→ 卡上 `bonsai_k1om_chat /tmp/bonsai.gguf chat`（`src/main_k1om.c` 的 chat 模式，
行协议 `E/R/P/G`，模型常驻、ctx=8192）。分词器硬测试向量
`"The capital of France is" == [760, 6511, 314, 9338, 369]` 通过；
中英双语实测通过（含思维链输出）。`--selftest` 可离线验分词器。

**chat 模式的卡侧特殊性（都踩过坑）**：
- `/tmp` 是 **tmpfs（ramfs）无 swap**：模型文件 + 内存副本双重驻留，
  2.6.38 的 OOM killer 不回收直接杀进程。→ **chat 模式加载后 unlink 模型文件**
  （loader 已全量拷进匿名内存并关闭 fd），再预触碰 1 GB KV。所以每次会话都要
  重新上传模型（chat.py 自动处理）。
- `system()`/fork 在 244 线程下等于自杀（~28 GB 虚拟栈被复制）→ 已用
  `kmp_set_stacksize_s(16MB)` 把 VmSize 降到 11.5 GB。
- Windows 宿主 text-mode 管道发 `\r\n` → 卡侧协议解析把 `\r` 当空白。
- **孤儿进程持有已删文件 = 7 GB 泄漏**：曾被一个残留 `wc` 坑过一次
  （kill 掉即回收）。排查：`ls -l /proc/[0-9]*/fd | grep deleted`。

**SCIF 传输**（`tools/scif_send.c` 宿主 + `tools/scif_sink.c` 卡侧）：
消息路径实测 **39 MB/s**（SSH 是 15.7），模型 186 秒。md5 校验过。
宿主库是 `C:\Windows\System32\uSCIF.dll`（LoadLibrary 运行时解析，无 import lib）。
注意 sink 是**一次性**的（收完一个文件就退出；端口 20500 残留会导致
下次 bind EINVAL——先 `pkill scif_sink`）。升级到 RMA 注册内存路径
（scif_register + scif_writeto，文档记 1-2 GB/s）尚未做，是下一步。

---

> 写给下一个接手的人（或下一个会话的我）。
> **请先读这份，再读别的文档。** 有些旧文档含已被推翻的结论，本文档标注了哪些可信。
>
> 最后更新：**串行 glue 并行化之后**（5.64x）。

---

## 0. 修复轮结果（三个子代理 + Lead）

### 性能优化（第 2–9 轮）—— 找到一条验证为精确的 1.14x 路径，**未集成**

#### ⚠️ 先说两个被推翻的结论（避免后人重复踩坑）

**1. "V7 = 1.37x" 是测量假象，不是真实加速。**
反汇编每个 128 权重块：
```
v1_row   16 vpermd, 8 vpsrlv,  8 vmulps, 8 vfmadd
v7_row    2 vpermd, 2 vpmulld, 2 vpsrlvd, 2 vpandd, 2 vmulps, 8 vfmadd
```
V7 的 4 个 m 迭代**算出逐位相同的向量**（索引只依赖 `l&3`），ICC 做了 **CSE**。
⇒ V7 定价的是一个只有 2 次 permute 的解码器 —— 而**正确的解码器必须做 8 次**
LUT gather（每 16 权重组一次）。**那个 1.37x 从未被真正测到。**

**2. V8/V9 的"根因定位"是错的。**
`LUT` **不是**"读字节 `(lane&3)`"：`permutevar(idx,lut)` = `lut[idx&15]` = **`(idx&3)-1`**，
**只用索引 dword 的低 2 位（只有字节 0）**。
真正原因：`PQ2_IMCI_LOAD16` 给 lane `l` 的**只有 `qs[l]`**，
而第 m 组需要 `qs[4m+l/4]`（住在 lane `4m+l/4`）；
`sllv`/`srlv`/`vpmulld` **全是 lane 本地的**，
**任何移位向量都无法完成那次广播** —— 穷举 4 种移位 × 200 随机块，
128 个三元组里 **126 个无解**（只有权重 0 和 64 例外）。

实测：`V9 (规范向量) 6016 失配 / V9u (统一 8*m) 6336 失配`。

**附**：我曾说 `_mm512_sllv_epi32` "不可链接" —— **错，它可链接（exit 0）**。
不可链接的是 `slli/srli/srai_epi16` 和 `mullo_epi16`（16 位版本）。
我把"`slli_epi32` 只接受标量立即数"（签名限制）和"链接失败"混为一谈了。

#### ★ V10：验证为精确的可行方案（未集成）

**按 2-bit 字段重组点积**，并把**输入 x 每次 matvec 调用置换一次**（而不是逐行置换权重）：

```c
xp[((b*2+h)*4+f)*16 + l] = x[128b + 64h + 4l + f];   // 恰好 n 个 float，无额外内存
```
内层：2×LOAD16 + 6×srli + 8 LUT permute + 8 mul + 8 FMA + 8 xp 载入
—— **8 次广播 permute 消失，没有增加任何东西。**

```
验证：V10 / V10b / V10c 全部 MISMATCHES = 0  *** EXACT ***
      64 种源对齐 × 129 检查 + 4 个随机块 × 8 对齐 + 多块 n=512（相对差 0.0e+00）
定价：V10 1.075x  /  V10c（批量调度）1.142–1.146x
      xp 构建 0.041–0.056 ms/次 @ n=5120（占 1.8% @ d=512）
真实收益：V1 1078 → V10c 945 ps/权重，差 133 ps/权重（不是原估的 ~350）
```

**未集成**：`pq2_0.h` 未被改动。集成需要在 `mm_folded` 里构建 `xp`，
并重跑端到端（top-1 **加上** `l_out-63` —— V10 改变了 FMA 配对顺序，
有 ~1e-7 量级的求和顺序差异）。

**文件**：`src/bonsai/pq2v9.c`（V10/V10b/V10c + 隔离探针）、
`src/bonsai/pq2v9_card_evidence.txt`（原始卡上输出）、`src/bonsai/_v9model.py`（穷举搜索）。

#### 其他被排除的路径

```
内存往返不是瓶颈              去掉 8 store + 8 load → 收益为零（933→932）
整数替代 LUT permute          更慢 2.1x，且数值错（V3）
extload 打包 + 按字节移位      前提不成立（extload UINT8 是每 lane 一个零扩展字节）
向量 popcount                 KNC 上【不存在】（_mm512_popcnt_epi32 编译期拒绝）
                              标量 _popcnt32 有且正确，但逐 lane 用会毁掉向量化
                              且 5 条运算替代 2 条，代数对但亏
```

**剩下的 8 次 LUT permute 无法消除** —— 逐 lane 转浮点只有 `vpermd` 查表这条路
（无 `cvtepi32_ps`），而 KNC 没有 `shuffle_epi8`/`maddubs`/`mullo_epi16`。

> **第 2–9 轮没有改动任何已验证代码** —— 核心源文件 md5 与修复轮结束时完全一致。

> **第 2–6 轮没有改动任何已验证代码** —— 核心源文件 md5 与修复轮结束时完全一致。

---

### 修复轮（第 1 轮）

| 成员 | 独占范围 | 结果 |
|---|---|---|
| `oracle-rope` | `scripts/bonsai_forward.py` | RoPE 配对 相邻→前后半，**全 64 层验证**（不是前缀） |
| `model-core` | `model.c`, `model.h` | 6 类缺陷全修，**10 个 BEFORE/AFTER 差异测试** |
| `entry-harness` | `main_k1om.c`, `test_forward.c` | 参数白名单、粘性退出码、**3 个新隐藏状态地标** |
| Lead | `gguf.*`, `pq2_0.h`, `docs/*` | 越界余量、hadamard 前置条件、文档修正、集成验收 |

**最终验收（全过）**
```
宿主 test_forward   ALL CHECKS PASSED（19 分钟，含 4 个隐藏状态地标 + 5 个阶段检查）
                     top-1 = 11751 MATCH, rank 0/248320, l_out-63 ratio 0.9997
oracle 自测          SELFTEST PASS
加载器自测           ALL CHECKS PASSED (0 failures)
k1om 端到端          k1om E2E PASS（top-1 MATCH, rank 0, 0.268 tok/s）
卡上参数校验          无参数/非法 mode/非法线程数/参数过多 → 全部 exit 2 + 明确报错
两平台编译           host exit 0（-Wall -Wextra 无警告）/ k1om exit 0
根目录               0 个陈旧副本
```

**关键度量：门控现在能失败了。** 用两个故意破坏的变体验证：
- **仅 RoPE 配对**错误 → top-1 仍然 **MATCH、rank 0**，但 `l_out-63` MISMATCH → exit 1
- **单层权重转置** → 4 个地标全部 MISMATCH，norm 比值 1.3062 → exit 1

**一条重要的实测发现**：RoPE 配对错误在**早期地标上几乎不可见**
（`l_out-0` 逐位相同、`attn_residual-3` 9.63e-4→1.34e-3、`l_out-3` 1.22e-3→1.27e-3），
**只在 `l_out-63` 才分离**（4.60e-2→1.01e-1）。**深度累积才是放大中段错误的机制**，
所以早期容差没有被调到那个抖动区间（那会变得脆弱）。

**已知最脆弱的一行**：`l_out-63` 的 first3 检查只有 **8.7% 余量**（4.60e-2 vs 5e-2 容差）。
若有人改动 `model.c` 的数值行为，先看这里。

---

## 1. 一句话现状

**Ternary Bonsai 2 27B 已能在 Intel Xeon Phi 7120P（k1om）上原生运行，输出与官方实现精确一致
（`top-1 = 11751 ' Paris'`，rank 0/248320），速度 0.267–0.289 tok/s。**

参考实现（宿主 + 卡）都验证过；性能是当前主要短板。

---

## 2. 立即要知道的五件事

### ① 权威版本在 `src/bonsai/`，根目录的旧副本已删除
根目录曾有 6 个陈旧副本，其中 `pq2_0.h`（228 行）**只含 4 次 `_mm512_fmadd_ps`**
—— 正是那个"结果恒为正确值一半"的 bug。实测：
```
从根目录执行 icc -mmic ... src/bonsai/main_k1om.c model.c gguf.c
→ 链接成功（exit 0），但产出的是半权重内核的二进制【静默地】
```
已删除，且当时确认无脚本引用。**别再重建它们。**

### ② top-1 / rank-0 **不足以**判定正确性
实测：`bonsai_forward.py` 的 RoPE 用**相邻配对**、`model.c` 用**前后半配对** ——
**两者都给出 top-1 = 11751、rank 0。**
⇒ **`MATCH` 不代表正确。** 必须同时看隐藏状态检查（`l_out-63`）。

### ③ 瓶颈是**计算**，不是带宽
```
每 token 需流 7.137 GB 权重
实测带宽（rows-1360 模式，正是内核的访问模式）= 196.43 GB/s  →  36 毫秒
实际每 token 耗时                                          = 3456 毫秒
→ 访存占 1%，计算占 99%
```
**带宽有 95 倍余量。** 不要再去优化访存。

### ④ k1om 是 **LP64**（`sizeof(long)==8`）
实测：在 mic0 上 `sizeof(long)=8`、`sizeof(off_t)=8`、`ftello` 能处理 3 GB 文件，
且**去掉** `-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE` 行为完全一致。
（早期文档说"32 位 long 是坑"是**错的**。）

### ⑤ KNC 上"头文件里有声明" ≠ "能链接"

**而且链接探测本身也有陷阱**：若内建函数的结果没被使用，
编译器会**死代码消除**掉调用，于是链接"成功" —— **假阳性**。

我因此**两次**误判：
- 第一次探测（结果存数组但没打印）→ 误判 5 个内建函数可用
- 用"把结果打印出来"的程序复核 → **全部链接失败**，与最初结论一致

**正确写法**：单宏编译 + **让结果可观察**（打印 / volatile / 影响返回值）。

```
【能链接】permutevar_epi32 (vpermd), permute4f128_epi32, swizzle_epi32,
          shuffle_epi32, alignr_epi32, slli/srli/srai_epi32, sllv/srlv_epi32,
          and/or/xor_si512, add/sub/mullo/abs/min/max_epi32,
          load/store_epi32, extload_epi32(NONE+UINT8),
          extloadunpacklo/hi_epi32, i32extgather_epi32,
          set1_epi8/16, set_epi32/64, setzero_si512,
          add/sub/avg/sad/max_epu8 (dvec.h)

【链接失败】_mm512_cvtepi32_ps (→不存在的 _mm512_cvt_roundepi32_ps),
          slli/srli/srai_epi16, mullo_epi16, madd_epi16, maddubs_epi16,
          mulhi_epi16, shuffle_epi8, alignr_epi8, alignr_epi64,
          shuffle_i32x4, permutexvar_*, permutex2var_*,
          bslli/bsrli_epi128, 整个 unpacklo/hi_epi8/16/32/64 家族,
          所有 *loadu*/*storeu* 形式
```

**任何新指令先用 `src/bonsai/pq2settle.c` 的模式做链接探测 + 数值核对。**

---

## 3. 构建与运行（已验证可用）

### 交叉编译到 k1om
```powershell
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'

Push-Location src\bonsai
& $icc -mmic -O2 -std=c11 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE `
       main_k1om.c model.c gguf.c -o D:\zigtest\k1om\bonsai -lm
Pop-Location
```
产出验证：ELF `machine = 0x00B5`（181 = `EM_K1OM`）。**不是** 0x3E。

### 宿主机（Zig 提供 clang）
```powershell
$zig = (Get-ChildItem D:\zig -Recurse -Filter 'zig.exe' | Select-Object -First 1).FullName
Push-Location src\bonsai
& $zig cc -O2 -std=c11 -o D:\zigtest\tf.exe test_forward.c model.c gguf.c
Pop-Location
```

### 部署到卡（两种通道）
```powershell
$key = "$env:USERPROFILE\.ssh\id_rsa_mic"
$kh  = "-o UserKnownHostsFile=`"$env:USERPROFILE\.ssh\known_hosts`" -o StrictHostKeyChecking=no"
```
**通道 A：SSH/scp（~13–16 MB/s，慢）**
```powershell
scp -O -c chacha20-poly1305@openssh.com -i "$key" $kh <exe> root@192.168.1.100:/tmp/
ssh -i "$key" $kh root@192.168.1.100 "chmod +x /tmp/<exe> && export LD_LIBRARY_PATH=/tmp && export OMP_NUM_THREADS=122 && /tmp/<exe> /tmp/bonsai.gguf forward"
```
`chmod +x` **必须**（BusyBox）。BusyBox 的 `grep` 不支持 `-E`，`head` 不支持 `-6`。

**通道 B：SCIF（`micnativeloadex`，快得多）**
```powershell
$ml  = 'C:\Program Files\Intel\MPSS\bin\micnativeloadex.exe'
$sys = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux'
$omp = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\compiler\lib\intel64_win_mic'
$env:SINK_LD_LIBRARY_PATH = "$sys\lib64;$sys\usr\lib64;$omp"
& $ml D:\zigtest\k1om\bonsai -a "/tmp/bonsai.gguf forward" -e "OMP_NUM_THREADS=122" -t 900
```
坑：变量名是 **`SINK_`**LD_LIBRARY_PATH；它**在宿主侧**解析依赖，所以要指向 k1om sysroot。

### 卡上现有文件
```
/tmp/bonsai.gguf      7,206,168,928 字节  md5 e119192844abba9e7821f6b0e23022a3
/tmp/libiomp5.so      1,324,555 字节
/tmp/bonsai_k1om      （需每次重新上传，重启会清空 /tmp）
```
**`/tmp` 是易失的**（`/tmp` → `/var/tmp` → `/volatile/tmp`，落在这张卡的 tmpfs 上），
**卡一重启就没了，模型要重传（约 7.2 分钟 @ 15.9 MB/s）。**

---

## 4. 当前架构（简洁版）

```
模型：Qwen3.8-27B 的三值压缩版（PrismML Ternary Bonsai 2 27B）
      64 层 = 48 个 GDN（Gated DeltaNet，线性注意力）+ 16 个全注意力（i%4==3）
      权重 = PQ2_0：34 字节/128 权重 = 2.125 bpw，共 7.137 GB
      401 个权重在 Hadamard 旋转基下（正向 x' = H·(s⊙x)）
      embedding 逆向：s ⊙ (H x)   ← 顺序与正向相反

代码：
  src/bonsai/gguf.c      GGUF 容器加载（7.2 GB，64 位偏移）
  src/bonsai/pq2_0.h     PQ2_0 解码（标量 + IMCI 向量化）+ matvec 内核 + Hadamard
  src/bonsai/model.c     前向传播
  src/bonsai/model.h     几何与 scratch 结构
  src/bonsai/main_k1om.c k1om 入口
  src/bonsai/test_*.c    测试
  scripts/bonsai_forward.py  numpy 参考（★ 注意：它的 RoPE 是错的，见下）
```

---

## 5. 已确认正确的部分（可以依赖，改动时不要破坏）

| 项 | 证据 |
|---|---|
| 参考实现端到端 | 宿主与卡上都 `top-1 = 11751 ' Paris'`、rank 0/248320 |
| 6 个阶段检查 | `embed`/`attn_norm`/`signed`/`qkv`/`conv`/`l_out-63` 全 OK |
| k1om 加载器 | `test_gguf` **41/41 通过**（宿主与卡上都是） |
| 向量化解码器 | 卡上 **705,536 次元素比较，0 失配**（全部 64 种源对齐 + 5000 随机块） |
| `bonsai_fp16_to_fp32` | **全部 65,536 个 half 值**与 numpy 逐位一致 |
| `aligned_f32` | k1om 与宿主各 **27,000 次分配**，簿记全对（UBSan 干净） |
| Hadamard | 与显式奇偶矩阵一致到 4.8e-7；H²=I |
| 四个宏组合 | `__MIC__` × `BONSAI_FORCE_SCALAR_PQ2_0` 各自选择正确，全部编译通过 |
| 性能 | 0.161 → 0.267 tok/s（IMCI 修复 + 向量化解包），正确性不变 |

---

## 6. 已知缺陷（按优先级）

### P0 —— 会让人"把对的改错"或"静默产出错误结果"

| # | 位置 | 问题 |
|---|---|---|
| **A** | `scripts/bonsai_forward.py:61-78` | **RoPE 用的是相邻配对 `(2i,2i+1)`，应为前后半配对 `(i,i+32)`。** 它自己的 docstring 和 `BONSAI2_FORWARD_SPEC.md:375` 都写"不是相邻配对"。**危险点：两种配对都给出 top-1 MATCH**，所以这个错误的 oracle 会诱导人去"修"正确的 `model.c` |
| **B** | 全项目的正确性门槛 | 只靠 top-1/rank-0。**必须加隐藏状态检查**（`l_out-63` 那类），否则 RoPE 级别的错误检测不到 |
| **C** | `model.c` 数值路径 | **从不检查 `t->type`**。`GGML_TYPE_PTQ1_0`(143) 在 `gguf.h` 里被赋予与 PQ2_0 相同的 34 字节块大小，实测能加载成功；`dequant_to_f32` 会拒绝，但 `mm_folded`/`embed_row` 不看类型 → 用 PQ2_0 码表解码 → **静默垃圾 logits** |

### P1 —— 崩溃/内存安全（畸形输入触发）

| # | 位置 | 问题 |
|---|---|---|
| **D** | `model.c:539` | `ssm_d_conv==1` 时 `(size_t)(ck-2)` 下溢 → 向缓冲区**之前** memcpy 40 KB。`calloc(0)` 返回非 NULL，`:362` 的检查通过 |
| **E** | `model.c:267, 479` | `value_dim / ssm_dt_rank` 和 `rep = vh/kh` 在几何检查（`:269-274`）**之前**执行，而检查里没有这两个字段 → 实测 SIGFPE |
| **F** | `model.c:170` | `mm_plain` 把调用者缓冲区直接送进 IMCI 内核 → 16 字节对齐时 SIGSEGV。它**未被调用但留着**，且**无宿主对应路径**，只在卡上炸 |

### P2 —— 健壮性/错误传播

| # | 位置 | 问题 |
|---|---|---|
| **G** | `model.c:314-360` | ~25 个分配只校验了 5 个。16 GB 卡 + 7.2 GB 模型，OOM 是现实风险 → NULL 解引用 |
| **H** | `model.c` 14+ 处 | `dequant_to_f32`/`mm_folded` 返回值全被忽略。张量缺失时**留下陈旧 `y`**（logits 保留上一 token 的值），而 `bonsai_forward` 仍返回 0 |
| **I** | `main_k1om.c:88-91` | 解码循环的 forward 失败被丢弃；`:97` 只要 `best==OFFICIAL_NEXT` 就返回 0 |
| **J** | `main_k1om.c:33` | `mode` 是**黑名单**（任何非 `"selftest"` 的字符串都跑 7.2 GB forward 路径）；文档写的 `[threads]` 参数**从未被解析** |

### P3 —— 文档矛盾（会误导后来人）

| # | 位置 | 问题 |
|---|---|---|
| **K** | `pq2_0.h:203` | 越界读写"约 96 字节"，**实测 48**（守卫页，全部 64 种对齐同阈值） |
| **L** | `gguf.h:26-28` | 注释说"mmap 只读、从不拷贝"，实际是堆读取。**若有人"修正"加载器去匹配注释，越界读就会变成活的 SIGSEGV** |
| **M** | 多个文档 | 仍写"IMCI 默认禁用"（实际是 k1om 默认启用）；`BONSAI_IMCI_PQ2_0` 是**幽灵宏**（宿主上 `knc` 与 `ref` 是同一地址，所以相关测试是"拿参考和自己比"，永远通过） |
| **N** | 多份旧文档 | 含已被推翻的结论：`PQ2_VECTORIZED_UNPACK.md` 里"越界读依赖数据"（实为测试程序堆越界）、`K1OM_BUILD_RECIPE.md` 里"32 位 long"（实为 LP64） |

### P4 —— 性能（非正确性）

```
当前：0.267 tok/s（244 线程）/ 0.289 tok/s（122 线程，已实测最优）
瓶颈：计算绑定。纯解码 4.11 秒/token，实际 3.75 秒 → 解码约占 110% 时间
根因：vpermd 是 KNC 的【微码指令】，解码器每 128 权重需 16 次
余量：带宽 95 倍（196 GB/s 只用了 36 毫秒/token）
未做的实验：bias-trick 为什么对负值出错（当时没查清就放弃，可能省掉一半 vpermd）
```

---

## 7. 每项修复的验收标准（不可妥协）

```
1. 宿主编译通过：  zig cc -O2 -std=c11 src/bonsai/{test_forward,model,gguf}.c
2. 宿主回归全过：  D:\zigtest\tf.exe D:\bonsai\...gguf  →  ALL CHECKS PASSED
                   且 top-1 = 11751、rank 0，【并且】l_out-63 first3_err OK
3. k1om 编译通过：  icc -mmic ... → exit 0，产出 machine=0x00B5
4. 卡上运行：      通过 SCIF 或 SSH 跑 forward → MATCH、rank 0
5. 不破坏已确认项：解码器 0 失配、fp16 逐位一致、tok/s 不下降
```

**特别注意**：只跑 top-1 不算验证（见 ②）。

---

## 8. 这张卡的关键实测数据（避免重复测量）

```
内存带宽（6.87 GB buffer，3 次取最好，244 线程）
  contiguous    62.45 GB/s
  rows-1360    196.43 GB/s   ← 推理内核的访问模式
  rows-340     103.54 GB/s
  interleaved   47.37 GB/s
  峰值（官方）  352 GB/s

线程数 vs 端到端（同一二进制）
   61 线程  0.273 tok/s
  122 线程  0.289 tok/s   ← 最优
  244 线程  0.269 tok/s

单线程吞吐（子代理实测）
  解码 10240x5120 (DRAM 流式)  589.7 M 权重/s
  解码 1024x32   (L2 驻留)    1254.4 M 权重/s
  matvec 10240x5120           0.4993 s → 0.1008 s（4.98x）

解码器越界读
  足迹 [blk, blk+81]，即最多 48 字节越过 34 字节块的末尾
  与源对齐无关（64 种相位同阈值）
  纯读，不产生错误数值（毒化那 48 字节，0/128 权重改变）
```

---

## 9. 硬件/环境

```
卡：Intel Xeon Phi 7120P (Knights Corner, k1om)
    61 核 / 244 线程 @ 1.238 GHz，15.5 GB GDDR5（ECC）
    Linux 2.6.38.8+mpss3.8.6，BusyBox
    mic0: online，IP 192.168.1.100，SSH: ~/.ssh/id_rsa_mic

宿主：Windows 10 Pro for Workstations
      PSXE 2017 (ICC 17.0.0) 在 C:\Program Files (x86)\IntelSWTools\
      MPSS 3.8.6 在 C:\Program Files\Intel\MPSS\
        └─ k1om sysroot 就在里面（k1om-mpss-linux\）
      Zig 0.17.0（提供宿主机 clang）在 D:\zig\
      无 MSVC（COI 的 C++ 样例不能用；但宿主侧 COI 可用 Python ctypes 驱动）

模型：D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf  7,206,168,928 字节
      md5 e119192844abba9e7821f6b0e23022a3
ground truth：D:\bonsai\eval_callback_full.txt（5-token dump，6.67 MB）
              D:\bonsai\dump_single.txt（单 token dump）
官方参考：D:\llama-prism-win\（46 个官方二进制）
fork 源码：D:\llama-prism-src\llama.cpp-prism\
```

---

## 10. 文档可信度分级

| 文档 | 可信度 |
|---|---|
| 本文件 | 最新，第三轮审查之后 |
| `CODE_AUDIT_RESULTS.md` | 三轮审查汇总，可信 |
| `FINAL_STATUS_K1OM.md` | 状态描述可信，但"预期速度"部分已过时 |
| `pq2_0_imci_decoder.md` | 解码器细节可信（含 0 失配证据） |
| `KNC_INTEGER_SIMD.md` | 指令可用性可信（以链接器为准） |
| `K1OM_TRANSFER_CHANNELS.md` | 通道数据可信 |
| `COI_ROUTE.md` | COI 现状可信 |
| `PERFORMANCE_OPTIMIZATION.md` | **⚠️ 基于错误的 3.32 GB/s，结论"带宽会挡住优化"是错的**，需重写 |
| `PQ2_VECTORIZED_UNPACK.md` | **⚠️ 中途记录，含已被推翻的假设**（"越界读依赖数据"实为堆越界） |
| `K1OM_BUILD_RECIPE.md` | 配方可用，但"32 位 long"那条是错的 |
| `GDN_AXIS_AND_L2_FINDINGS.md` | 历史记录，结论已被后续修正 |
| `BONSAI2_K1OM_PLAN.md` | 早期计划，多处过时 |

---

## 11. 接手建议

```
1. 先读本文件，再读 CODE_AUDIT_RESULTS.md
2. 按 P0 → P1 → P2 → P3 修（P0 的第一项是修 oracle 的 RoPE）
3. 每修一项走第 7 节的验收流程，不要只看 top-1
4. 性能优化（P4）放到最后 —— 在可能不正确的实现上优化没有意义
5. 善用子代理：这次的经验是【范围要小】。一个覆盖 6 文件 + 9 关注点的审查
   任务会让代理静默退出；拆成 2-3 个文件一组 + 明确验收标准才有效
```

## 12. 我犯过的错（供后来人避免）

| 错误 | 教训 |
|---|---|
| 带宽报错 59 倍（1 GB buffer 命中缓存 + 计时器分辨率） | **测量要用足够大的工作集，并怀疑自己的基准** |
| 看到 104 GB/s 却判定"是计时假象" | **不要因为数字"太大"就丢弃测量结果** |
| 用 top-1 当正确性判据 | **"通过"不等于"正确"** —— 要针对性地设计能证伪的检查 |
| 查 IMCI bug 时验证中间量而非结果 | **断言「结果 == 参考」，不是「中间量 == 假设值」** |
| 没验证指令可用性就写代码 | **KNC 上先链接探测** |
| 误判"PSXE 未安装" | **搜索命令失败 ≠ 文件不存在** |
| 误判 k1om 是 32 位 long | **文档里的旧笔记也会错，要实测** |
| 拿 6 个采样点 sqrt 折算高维向量 | 高维张量的量级不能靠采样外推 |
| 把 GB/s（打包字节）当成 G 权重/s，虚构出"4.3x 内核差距" | **换算前后单位写进数字旁边**；2.125 bpw 意味着两者差 3.76 倍 |

---

## V12 集成验收（已完成）

V12 解码器（40 → 30 ops/128 权重）已集成进 `src/pq2_0.h`，**完整验收通过**：

```
宿主回归   ALL CHECKS PASSED，exit 0（18.3 分钟）
           5 个阶段检查 + 4 个隐藏状态地标全部 OK
           l_out-63 first_err=4.60e-02（与 V12 之前【完全相同】）
           ||x|| ratio = 0.9997，top-1 = 11751 MATCH
k1om 编译  exit 0
卡上运行   decode 1.717 tok/s，top-1 11751 MATCH，rank 0/248320
```

**V12 的求和顺序差异未影响任何地标** —— 实测与集成前逐位同值。

性能历程（全部卡上实测）：
```
0.292 tok/s  原始
0.962        + DeltaNet 递推 head-parallel
1.257        + Hadamard 块并行
1.407        + conv 通道 + 逐头 L2 norm
1.644        + 默认线程 122 -> 244
1.711        + V12 解码器
             = 5.86x
```

**集成方式**：`build_xp` 放在 kernel 内部、用线程私有栈数组
（避免共享缓冲区的数据竞争，且不改函数签名）。
旧内核保留在 `pq2_0.h` 的 `#if 0` 里作参考。
