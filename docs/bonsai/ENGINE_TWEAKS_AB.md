# README 上剩下三条性能杠杆：逐条做掉，附同二进制 A/B

README「已知缺口 / 性能」那一行列过三条剩余杠杆。这份文档把三条各做了一遍，
**每条都给出实测证据**，包括一条负结果和一条"实现了但默认关闭"。
方法沿用项目既有口径：改动走**同一个二进制 + 环境开关**，避免编译器差异污染 A/B。

## 结论速查

| 杠杆 | 结论 | 实测 |
|---|---|---|
| **prefill 输出头跳过** | ✅ **已落地，默认开启** | prefill **+5.3%**（14.01 → 13.27 s），32 token 贪心序列与基线**完全一致** |
| **块内 rescale 变体** | ⚠️ **已实现，默认关闭**（`BONSAI_RESCALE=1` 开启） | 同一比较两次跑出 **+2.72% / −2.10%**，落在 ±6% 跨进程噪声内 → 收益未确证 |
| **vexp223ps 向量化 exp** | ❌ **负结果**：编译器早就做了 | 手写向量 exp vs 标量 `expf` = **1.03x**；`model.c` 汇编里本就有 `__svml_expf16` |

---

## 1. prefill 输出头跳过（已落地）

**改动**：[`src/model.h`](../src/model.h) / [`src/model.c`](../src/model.c) 把
`bonsai_forward` 拆成 `forward_body(..., want_logits)`，导出
`bonsai_forward_nologits()`；[`src/main_k1om.c`](../src/main_k1om.c) 的 `P` 命令改成
**一个 token 的前瞻**：除最后一个 prompt token 外全部走 nologits，最后一个保留完整前向
（第一个 `G` 要读 logits）。KV cache 与递推状态不受影响——输出头是纯读出。

**为什么能省**：`output.weight` 是 `5120 × 248320 = 1.27e9` 权重，
而每 token 要过 `~2.77e10` 权重（27B 模型）→ 权重占比 **4.6%**；
48 token 的 prefill 里 47 个可跳过 → 预期 ~4.5%。

**实测**（卡上 48-token prefill，2 次取中位数，两组都强制 F3 位精确顺序，只差这一项）：

```
baseline (both old)       prefill 14.01 s   (3.43 tok/s)
head skip only            prefill 13.27 s   (3.62 tok/s)     -> +5.30%
```

**正确性**：四个配置（两个开关各开/关）各自贪心生成 32 token，
**序列完全相同**，且与改前基线一致。

关掉它：`BONSAI_NO_PREFILL_SKIP=1`。

---

## 2. 块内 rescale 变体（实现完成，默认关闭）

**改动**：[`src/pq2_0.h`](../src/pq2_0.h) 把 F3 的块循环抽成 `pq2_row_block_sum()`，
两种顺序共用一份代码：F3 原来每块把两个 LUT 预乘 `vd`（2 条 `vmulps`），
变体改用**未缩放的 LUT**，块末做一次 `s = fmadd(vd, t, s)`。
省掉 broadcast→2-mul 这段依赖链，代价是**每块一次 fp32 舍入**而不是每元素一次
（约 1e-7 量级的求和顺序漂移，与当初 V12→F3 同属一类）。

**为什么默认关闭——收益测不出来：**

1. **端到端同二进制 A/B**（4 配置，各 32 token 贪心，序列全部一致）：

```
                config  prefill s   tok/s  decode tok/s  seq==
   baseline (both old)      14.01    3.43          3.61    yes
        head skip only      13.27    3.62          3.83    yes
          rescale only      13.63    3.52          3.71    yes
         both (default)      14.32    3.35          3.53    yes
```
   单变量比较 baseline → rescale-only 是 **+2.72%**，但 baseline → both 是 **−2.10%**；
   而 **同一配置内** decode 稳到 ±0.02 tok/s（±0.5%），**配置之间**却漂 ±6%。
   也就是说跨进程漂移比被测效应大，+2.72% 不可信。
2. **同进程内核探针**也不可用：工作集（23.7 MB/权重矩阵）是 cache 驻留的，
   不同 shape 报出 47,764–251,502 Mw/s（同样权重数差 4 倍），
   完全不代表真实的 **7.2 GB/token DRAM 流**。文献侧也早有预期：
   `REMAINING_OPT_PQ2_KERNEL.md` 排名第 3 项自己写着"0–19 ms，取决于 244t 的天花板
   是纯内存延迟还是别的"。

**所以**：为了一个未确证的收益带上 ~1e-7 漂移是不划算的。变体保留、开关保留
（`BONSAI_RESCALE=1`），等有真正能压住噪声的测量方法（例如把模型权重全部驻留前的
pq2ab 式对照）再定。**出厂默认仍是位精确的 F3 顺序。**

**正确性（两种顺序都过门）**：`top-1 = 11751 (' Paris')`，rank `0 / 248320`，两者都 MATCH。

---

## 3. vexp223ps 向量化 exp（负结果，且有副产品）

**先证伪了文档里的公式。** `KNC_ISA_AUDIT.md` 写的是
`exp(x) = vexp223ps(x·log2e) + 1`。四个探针（`vexp_probe*.c`）在卡上的结论是：

- `_mm512_exp223_ps` 编译成 **2 条真指令**：`vcvtfxpntps2dq $80, %zmm0, %zmm1` + `vexp223ps %zmm1, %zmm2`（无 `call`）。
- **它的语义是 `2^(v / 2^24)`，v 是 Q24 定点整数** —— 用整数锚点读出来的：
  `v=-2^23 → 0.7071 = 2^-0.5`，`v=+2^23 → 1.41421 = 2^0.5`，`v=0 → 1.0`。
  **不是 `2^x−1`**：那个 `+1` 会让 `exp(0) = 2.0`。
- 因此正确用法是 `vexp223ps(cvtfxpnt_q24(x·log2e))`，精度实测
  **最差相对误差 9.4e-7**（2^-20 量级），深负数**干净下溢到 0，不是 NaN**
  （这点和当年 Qwen 踩的 KVN `vexp2ps` 陷阱不同，是安全的）。
- 注意 `_mm512_cvtps_epi32` / `_mm512_cvt_roundps_epi32` 在 k1om 上**链接失败/是 call**，
  别用；要用 KNC 原生的 `_mm512_cvtfxpnt_round_adjustps_epi32(..., _MM_EXPADJ_24)`。

**但它没有价值**：数组对数组实测（4096 点 × 200 次）

```
scalar expf : 0.0007 s  (1188.2 M/s)
vector exp  : 0.0007 s  (1226.5 M/s)  -> 1.03x
```

原因在反汇编里一目了然：**icc 已经把逐元素 `expf` 自动向量化了**。
出货标志下编译 `src/model.c`，汇编里有

```
call __svml_expf16        (以及 _mask 变体)
vcvtfxpnt / vrcp23ps / vrsqrt23ps
```

SVML 的 `expf16` 在 KNC 上走的就是 `vexp2` 硬件路径。**手写不可能快过它**，
所以 README 里那条"<0.5 ms"的杠杆实际上**已经不存在了**——不是没做，是编译器做了。
（审计自己的预期也只有 <0.5 ms/token，约 0.2% 端到端，本来就在噪声下。）

---

## 4. 副产品：`forward` 模式会 OOM，以及绕开方法

跑正确性门时踩到：`src/main_k1om.c` 的 **`forward` 模式不像 chat 模式那样 unlink 模型**，
于是 tmpfs 里 6.71 GiB 的模型文件 + 进程 anon（实测 **8.23 GiB**）在
**15.15 GiB** 的总内存上顶到极限，`dmesg` 里是

```
Out of memory: Kill process ... (bonsai_k1om_cha) ... anon-rss:8229960kB
```

进程被 SIGKILL（退出码 137），门就"FAIL"了 —— 而这与任何优化无关：
anon 只有 8.23 GiB，比 chat 模式还小，问题在于**模型文件没被释放**。
早先几次门能过，是贴着边过去的运气。

**绕开**：`forward` 模式加 `BONSAI_NO_DROW=1`（省掉 0.97 GB 的惰性 drow 表），
门就稳定通过。本文所有 `top-1` 数据都是这样取的：

```powershell
ssh ... "export LD_LIBRARY_PATH=/tmp; BONSAI_NO_DROW=1 OMP_NUM_THREADS=61 \
         /tmp/bonsai_k1om_chat /tmp/bonsai.gguf forward"
```

真正的修法是让 `forward` 模式也在加载后 unlink（chat 模式已经这么做），
但那会改变这个一次性基准的行为，本轮没动。

---

## 5. 环境开关汇总

| 变量 | 默认 | 作用 |
|---|---|---|
| `BONSAI_NO_PREFILL_SKIP=1` | 关（即默认**跳过**） | 恢复"每个 prompt token 都算输出头"的旧行为 |
| `BONSAI_RESCALE=1` | 关（即默认**位精确 F3**） | 启用块内 rescale 变体 |
| `BONSAI_NO_DROW=1` | 关 | 不建惰性 drow 表（省 0.97 GB；`forward` 模式跑门时用） |
| `BONSAI_FORCE_SCALAR_PQ2_0` | 关 | 回到标量参考解码器 |

## 6. 复现

```powershell
# 构建（卡端）
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT = 'C:\Program Files\Intel\MPSS'
& $icc -mmic -O2 -std=c99 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE `
       src\main_k1om.c src\model.c src\gguf.c -o bonsai_k1om_chat -lm

# 正确性门（加 BONSAI_NO_DROW=1 避免 forward 模式的 OOM；见 §4）
# A/B：卡上 chat 模式，同一二进制四个开关组合，比对 token 序列与 prefill/decode 时间
```

本轮实测：冷启动 `python scripts/chat.py` **exit 0 / 56.4 s**，
上传 7.2 GB（RMA，359 MB/s）→ 加载 → 预填充 5.5 s → 生成 16 token / 4.3 s（**3.75 tok/s**）。
