# 作业说明：k1om 推理的真正瓶颈 —— parallel region 开销

> ## ❌ 已结案：假设被实测否定（后续轮次）
>
> **parallel region 开销不是瓶颈。** pq2reg：401 次 region × 34 µs = 13.6 ms/token
> （占 matvec 5%）；pq2call：5 次真实 shape 独立调用（99.06 G w/s）≈ 1 次大调用
> （90.68 G w/s），A≈B。
>
> **而"模型内 25.9 G w/s vs 独立 111.65 G w/s 的 4.3x 差距"是单位错误**：
> 25.9 是 GB/s（打包字节），×128/34 = **97.5 G 权重/s**，与独立基准在真实
> shape 混合下的 99.06 G w/s 一致。**内核在模型里一直跑在满速。**
>
> 后续实际有效的优化见 `HANDOVER.md` §0c（1.711 → 2.686 tok/s）。
> 本文档其余部分仅作历史记录。
>
> ---

> **给下一个执行者。** 这份说明修正了之前 10 轮工作的方向。
> 请先读 `HANDOVER.md` 和本文件，**再**读 `docs/PQ2_DECODER_COST_ANALYSIS.md`。
>
> 最后更新：第 11 轮之后。

---

## 1. 一句话结论

**瓶颈很可能不在解码器，而在"每 token 数百次 OpenMP parallel region 的进入/退出"。**

之前 8 轮全部在优化解码器的指令数（V3/V7/V8/V9/V10/V12/F3），
那些工作本身是对的，但**它们作用的只是端到端时间的一部分**。

---

## 2. 已经确立的事实（可信）

| 事实 | 数值 | 来源与可信度 |
|---|---|---|
| 当前性能 | **0.268 tok/s**（244 线程）/ 0.290（122） | 端到端实测，多次 |
| 每 token 时间 | **3.73 秒** | 同上 |
| matvec 占比 | **81.8%** | 用模型**真实张量形状**逐张量加总（3051/3730 ms） |
| 卡上生效的内核 | **`pq2_0_matvec_knc`**（IMCI） | `icc -mmic -E model.c` 第 15835 行确认 |
| 空 parallel region 开销 | **4006 us @122 线程** | `pq2omp.c` 卡上实测，**有 barrier，有效** |
| 每 token matvec 调用次数 | **401 次** | 64 层×5 + 48 + 32 + 1（**需实测确认**） |
| 实测每调用时间 | 3730 / 401 = **9.30 ms/次** | — |
| 关键矛盾 | 内核单独测看起来很快，但模型里 401 次调用共 3.05 秒 | 见第 4 节 |

### 硬件事实

```
61 核 / 244 线程 @ 1.238 GHz，15.5 GB，L1 32 KB，L2 512 KB/核
实测流式带宽（内核的访存模式）196 GB/s
每 token 只需流 7.14 GB ⇒ 带宽只需 36 毫秒（占 1%）
⇒ 带宽不是限制
```

### 发射率实测（有效）

```
每线程恰好 1 条 512 位向量操作/周期（与 ILP 无关：1→8 条独立链吞吐不变）
每核 4 线程 = 3.98 ops/cycle
⇒ 【每条 512 位指令成本相同】⇒ 唯一货币是指令数
```

---

## 3. ⚠️ 无效的测量 —— 不要采信，也不要重复这个错误

**`src/bonsai/pq2budget.c` 的 `matvec_knc` 数字（101.7 G w/s @122 线程）是假的。**

原因：循环里用**完全相同的参数**反复调用内核 ⇒ 编译器合并/提升了调用，实际只算了一遍。

**同一个文件还有一个编码问题**：我曾用 PowerShell `Set-Content -Encoding UTF8` 改过它 ——
那正是这个项目反复警告的隐患（会引入 BOM / 改坏编码）。

**另外 `pq2issue.c` 的 mix 数字也可能受影响**（`chains` 内层循环体相同，可能被合并）。
它的 `chains` 不变性结论仍然可信（那是"没有变化"的证据，不是"快"的证据），
但绝对吞吐数字不要引用。

**⇒ 本说明第 2 节里，凡是来自这两个文件的数字我都已剔除。**
`pq2budget` 只有 **`matvec_ref` 的缩放曲线**勉强可用（9.8 G w/s @122），但也需要重测确认。

---

## 4. 核心矛盾（要解决的）

```
内核单独测（一次调用、开销不摊销）：看起来很快
模型里 401 次调用：                  3.05 秒 = 9.30 ms/次

而 OpenMP 空 region @122 线程：      4.0 ms
⇒ 若每次调用都付这个代价：4.0 × 401 = 1.60 秒 = 每 token 的 43%
```

**这个假设能解释一系列观察**：
- 为什么模型里所有张量都约 8.8 G w/s（**恒定开销主导**，与张量大小无关）
- 为什么并行只有 7.8x（区域开销本身随线程数增长）
- 为什么解码器优化（F3 的 1.84x）在端到端可能只兑现一小部分（**它不减少区域数**）

**但这是假设，不是结论。** 需要实测判定。

---

## 5. 第一步（最重要，且很便宜）：实测每 token 的 parallel region 数

**不要相信 401 这个估算** —— 它是我按张量清单算的，可能有遗漏或重复。

**做法**：在 `mm_folded` 里放一个计数器（`model.c` 的副本，不要改原文件）：

```c
/* 在 mm_folded 开头 */
#ifdef BONSAI_COUNT_REGIONS
    extern volatile long g_mm_calls;
    __sync_fetch_and_add(&g_mm_calls, 1);
#endif
```

跑一个 5-token forward，打印总数，除以 5。
**同时打印每次调用的 d（行数）** —— 这能立刻看出是否有大量"很小的调用"。

**验收**：得到每 token 的**实测** region 数，以及行数分布。

---

## 6. 第二步：用**正确写法**重测内核缩放

**必须避免第 3 节的错误**：
```c
/* ❌ 错：参数不变，编译器会合并 */
for (int r = 0; r < reps; ++r) kernel(y, x, w, n, d);

/* ✅ 对：每次改变一个输入，并校验结果用于 sink */
for (int r = 0; r < reps; ++r) {
    w[0] = (uint8_t)r;                     /* 破坏 CSE */
    kernel(y, x, w, n, d);
    sink += y[r % d];                      /* 让结果可观察 */
}
```
或者用 `_mm_mfence()` / `volatile` 指针阻断优化。

**要测的**：
- 内核在**真实张量形状**（5120→17408 等）下的缩放曲线：1/2/4/8/16/30/61/122/244 线程
- 对比：单个大调用 vs **401 个真实大小的小调用**（后者才是模型的实际形态）
  —— 这一条直接验证第 4 节的假设

**验收**：得到"401 次小调用"与"一次大调用"的吞吐差。若差很多，假设成立。

---

## 7. 第三步：如果是区域开销，怎么修

**修法：把每层的多个 matvec 合并进一个 parallel region。**

现在（每个 matvec 一个区域）：
```c
static inline void pq2_0_matvec_knc(...) {
    #pragma omp parallel for schedule(static)      /* ← 每次调用建一个区域 */
    for (int i = 0; i < d; ++i) { ... }
}
```

改成（每层一个区域，内部用 for）：
```c
#pragma omp parallel
{
    #pragma omp for schedule(static)               /* 第一个 matvec */
    for (...) { ... }
    #pragma omp for schedule(static)               /* 第二个，复用同一组线程 */
    for (...) { ... }
    ...
}
```

**但这里有个设计问题要先想清楚**：模型里 matvec 的调用是**散落在 `block_gdn`/`block_attn`/`block_ffn` 内部**的，
而且**中间夹着串行工作**（norm、softmax、GDN 递推）。所以不能简单地把它们都塞进一个区域 ——
除非把那些串行工作也搬进去（或接受区域内的 barrier）。

**三个可选方案，按侵入性排序**：
1. **零代码改动**：试 `OMP_WAIT_POLICY=ACTIVE` / `KMP_BLOCKTIME`
   —— ⚠️ **我试过，卡上 OOM 了**（自旋线程吃内存，被 OOM Killer 杀掉）。
   可以试**更低线程数**（如 61），或先调大 overcommit。
2. **中等**：把每层内的 matvec 用 `nowait` 或手动分区合并
3. **较大**：重构为"每层一个 parallel region，内部含所有 matvec 和串行段"

**验收**：端到端 tok/s 提升，且 top-1 仍为 11751 MATCH、`l_out-63` 仍 OK。

---

## 8. 独立性问题：F3 解码器优化（已验证 1.84x）

`src/bonsai/pq2f3.c`（子代理产出）有一个**验证为精确**的更快解码器：

```
F3 + B2:  1689 M w/s  (1.84x over V1 的 917.8)
验证: 33024 项元素检查，0 失配；多块（含奇数块数）相对差 0.000e+00
```

**但它减少的是指令数，不减少 region 数。** 如果第 4 节的假设成立，
它的端到端收益会被区域开销稀释。

**建议顺序**：先做第 5、6 步确认瓶颈，**再**决定 F3 的集成优先级。

---

## 9. 硬件与工具链（已验证）

```
交叉编译（从 src\bonsai）：
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
& $icc -mmic -O2 -std=c11 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE `
       main_k1om.c model.c gguf.c -o D:\zigtest\k1om\out -lm

部署 + 运行：
$key = "$env:USERPROFILE\.ssh\id_rsa_mic"
$kh  = "-o UserKnownHostsFile=`"$env:USERPROFILE\.ssh\known_hosts`" -o StrictHostKeyChecking=no"
scp -O -c chacha20-poly1305@openssh.com -i "$key" $kh <exe> root@192.168.1.100:/tmp/
ssh -i "$key" $kh root@192.168.1.100 "chmod +x /tmp/<exe> && export LD_LIBRARY_PATH=/tmp && /tmp/<exe> /tmp/bonsai.gguf forward 122"

模型在卡上 /tmp/bonsai.gguf（7.2 GB，md5 e119192844abba9e7821f6b0e23022a3）
卡 /tmp 是易失的（符号链接到 /var/tmp）
chmod +x 必须；BusyBox 的 grep 不支持 -E；head 不支持 -6
```

---

## 10. 这个项目的六条陷阱（每条都真实发生过）

1. **死代码消除**：结果不被使用 ⇒ 编译器删掉调用 ⇒ 链接/基准"成功"但什么都没测
2. **编译器 CSE**：变体的多个迭代算出相同结果 ⇒ 被合并 ⇒ 时基无效（"V7 = 1.37x" 就是这么来的）
3. **参数不变的循环调用** ⇒ 被提升 ⇒ 无效（本说明第 3 节，我犯的）
4. **跨二进制比较不可信**：同一源码在两个二进制里差 20%，必须放同一二进制、正反两遍
5. **"看起来成功"≠"真的成功"**：top-1 MATCH 但 RoPE 是错的；链接"成功"但符号不存在
6. **PowerShell `Set-Content` 会破坏编码**：只用 `read`/`edit`/`write` 工具改文件

**断言「结果 == 参考」，不是「中间量 == 假设值」。** 逐元素验证，0 必须是实测的 0。

---

## 11. 第一步的验收标准

```
1. 实测每 token 的 parallel region 数（不是估算的 401）
2. 实测"401 次小调用" vs "一次大调用"的吞吐差
3. 端到端 tok/s（若做了修复）且 top-1 = 11751 MATCH、l_out-63 OK
```

**如果第 2 步显示没有显著差异**，那我的假设是错的，瓶颈在别处 ——
那种情况请明确报告，并回到"用正确写法重测内核真实吞吐"。

---

## 12. 明确：本说明没有声称的东西

- **没有**声称区域开销就是瓶颈（那是待验证的假设）
- **没有**给出修复后的预期收益数字（我不知道）
- **没有**声称 F3 的端到端收益（它的 1.84x 是单线程内核级的）
- 第 2 节表格里标"需实测确认"的项，就是还没确认的
