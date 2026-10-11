# KNC 上 Qwen2.5 k_bias 导致 NaN 的根因分析与修复

> 日期：2026-08-02
> 目标平台：Intel Xeon Phi 7120P（k1om，61 核，16GB GDDR5）
> 结论先行：**这并非 KNC 硅层精度限制，而是 run.c 中 MIC softmax 的两处软件缺陷**。已修复并在硬件上验证。

---

## 1. 背景与症状

Qwen2.5-1.5B-Instruct 的中文适配中，`forward_mic_q80` 需要给 Q/K/V 投影加上 bias（Llama2 没有 bias，原 llama2.c 派生代码完全不处理）。应用 `s_k[i] += k_bias[...]` 后，**所有 logits 变成 NaN**。

早期排查（见 `docs/QWEN_CN_STATUS.md` 旧版）排除过：数据损坏、文件偏移、mmap 对齐、OpenMP 并发、编译器优化、浮点精度、循环结构。当时基于以下事实得出了**错误结论**：

- 只加前 3 个 k_bias 元素（值 4.1/-0.5/-1.8）→ 正常
- 加 128+ 个元素 → NaN
- k_bias 值域 -316 ~ +288（比 q_bias 的 -26~+29 大一个数量级）

于是判定为 "KNC 的 VPU（vexp2ps + _mm512_fmadd_ps）对大动态范围浮点值的精度不足，是硅层硬件限制"，并用 `#ifndef __MIC__` 在 KNC 上跳过 k_bias。

---

## 2. 关键技术背景

### 2.1 KNC 硬件指数指令 vexp2ps

`_mm512_exp_ps`（SVML）在 KNC 上最终映射到硬件指令 `vexp2ps`，计算 2^x。其实现为**范围归约**：

```
2^x = 2^floor(x) × 2^fraction
      │                  └─ 分数部分查表(LUT) + 多项式
      └─ 整部直接加到 FP32 结果的指数字段
```

这带来一个关键限制：**当 2^x 的结果超出 FP32 可表示范围时，整部加到指数字段会回绕**，产生 garbage/NaN，而不是干净的溢出(inf)或下溢(0)。

- FP32 最小正规数 ≈ 2^-126，次正规数下界 ≈ 2^-149
- 当 exp 输入（乘以 log2e 后）小于约 **-103** 时，结果低于 2^-149，`vexp2ps` 开始产生 NaN

### 2.2 MIC softmax 向量化（DEBUG_LOG 12.14）

softmax 的 max 归约、exp、求和、归一化均用 `_mm512_*` 向量化：

```
max_val:   _mm512_load_ps → _mm512_max_ps → _mm512_reduce_max_ps
exp:       x - max → _mm512_exp_ps
sum:       _mm512_reduce_add_ps
normalize: _mm512_div_ps
```

---

## 3. 根因分析

通过 DBG 插桩（逐层打印 NaN 位置与值）在 7120P 上实测，关键证据：

```
DBG_VAL l=0 q_bias=0.184 -0.262 ... k_bias=4.125 -0.488 ... v_bias=0.044 ...
DBG_VAL l=0 s_q=0.329 -0.506 ... s_k=5.520 -1.722 ... s_v=-0.077 ...   ← 全部有限、干净
DBG_NAN post_softmax_att layer=0 pos=0 idx=0                            ← 但 softmax 输出 NaN
```

layer0 pos0 的 scores 全部有限，**softmax 却输出 NaN**——NaN 由 softmax 自身产生。

### 根因 1：max_val 读到未初始化内存（NaN 的直接来源）

MIC softmax 的 max 归约：

```c
__m512 max_val_load = _mm512_load_ps(x);      // 一次读 16 个元素
int size_aligned = size / 16 * 16;             // size < 16 时为 0
for (int i = 16; i < size_aligned; i += 16) { ... }
float max_val = _mm512_reduce_max_ps(max_val_load);   // 对 x[0..15] 取 max
```

问题：**pos 0..14 时 `size = pos+1 < 16`**，第一个 `_mm512_load_ps(x)` 读到的 `x[size..15]` 是上一次 softmax 留下的**垃圾值**。一旦垃圾含 NaN（att 缓冲区是复用的，任何一次 softmax 产出 NaN 就会残留），`_mm512_reduce_max_ps` 得到 NaN → `expf(x[0]-NaN)=NaN` → 整层输出 NaN → 级联到所有层。

**为什么只有 k_bias 触发？** 因为该 bug 依赖 `x[16..]` 的垃圾内容。k_bias 把各 head 的 score 值域从 ±9.4 放大到 ±324，改变了垃圾值的分布，恰好把残留的 NaN 触发出来。这解释了 "加前 3 个 bias 正常、加 128+ 个才崩" 的阈值现象——前 3 个 bias 时 score 变化小，垃圾仍是旧的有限值。

> 该 bug 对所有模型、所有生成过程的前 15 个 pos 都潜在存在，只是之前垃圾恰好不含 NaN。

### 根因 2：vexp2ps 对超大负数输入的越界 NaN（pos≥16 时触发）

修复根因 1 后（pos≥16 时 max_val 正确），还需面对 k_bias 带来的 score 值域问题。定量分析（本地 safetensors 权重，FP64 对照）：

| 指标 | 无 k_bias | 有 k_bias |
|---|---|---|
| \|k\| max | 3.36 | 317.5 |
| score 值域 | -1.8 ~ 7.6（范围 9.4） | 233 ~ 557（范围 **324**） |
| softmax exp 输入 (x-max) | 约 -14 | 约 **-467**（乘 log2e 后） |
| vexp2ps 是否越界 | 安全 | **远低于 2^-149，产生 NaN** |

pos≥16 时，`exp(x-max)` 的最小输入达到 -324（e 空间）→ 2^-467（2 空间），远超 vexp2ps 次正规下界 → NaN。这是**独立于根因 1 的第二个 bug**。

---

## 4. 修复内容（src/run.c）

### 修复 A：max_val 只在有效元素内计算

```c
int size_aligned = size / 16 * 16;
float max_val;
if (size_aligned >= 16) {
    __m512 max_val_load = _mm512_load_ps(x);
    for (int i = 16; i < size_aligned; i += 16)
        max_val_load = _mm512_max_ps(max_val_load, _mm512_load_ps(x + i));
    max_val = _mm512_reduce_max_ps(max_val_load);
    for (int i = size_aligned; i < size; i++) if (x[i] > max_val) max_val = x[i];
} else {
    max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
}
```

### 修复 B：exp 输入钳位

softmax（输入天然 ≤ 0，只需下限）：

```c
__m512 exp_min = _mm512_set1_ps(-80.0f);
v = _mm512_max_ps(v, exp_min);   // 低于 e^-80 的项归一化后本来就 ≈ 0
v = _mm512_exp_ps(v);
```

FFN sigmoid（3 处，±88 上下限，保留 silu 的渐近行为）：

```c
sigmoid_val = _mm512_max_ps(sigmoid_val, _mm512_set1_ps(-88.0f));
sigmoid_val = _mm512_min_ps(sigmoid_val, _mm512_set1_ps(88.0f));
sigmoid_val = _mm512_exp_ps(sigmoid_val);   // exp(±88) 仍在 FP32 有限范围内
```

### 修复 C：KNC 重新启用 k_bias

去掉 `#ifndef __MIC__` 条件编译，Q/K/V 三个 bias 在 KNC 上全部应用。

---

## 5. 验证

### 5.1 方法

在 `forward_mic_q80` 各关键点插入 NaN 检测（bias 应用后、softmax 前后、最终 logits），打印 `layer/pos/idx` 定位第一个 NaN 出现的位置。构建含 DBG_LOGITS 的版本对比首步 logits 与 HF 参考。

### 5.2 结果

- **DBG_NAN 全链路 0 个**；layer0 s_q/s_k/s_v 值干净（s_k[0]=5.52 含 k_bias 4.125，确认 k_bias 已生效）
- **首 token：DBG_LOGIT 3837 = 12.6158**，与 HF transformers 参考（3837=12.59）一致
- 中文生成连贯，~18-22 tok/s（244 threads）

### 5.3 回归与性能（7120P 实测）

**stories110M（FP32，greedy，seed 42）**：新旧二进制输出**逐字符一致**，tok/s 95.8 → 97.7，无回退。

**Qwen2.5-1.5B Q8_0 线程扩展**：

| 线程 | tok/s |
|---|---|
| 4 | 2.36 |
| 16 | 6.54 |
| 61 | 8.70 |
| 244 | 22.17 |

**chat 模式**（qwen2 模板）：正常，需管道 stdin（交互式多轮）。

---

## 6. 结论与教训

1. **"KNC 硅层精度不足" 的旧结论是错误的**。k_bias NaN 是两个软件 bug：softmax 越界读内存 + vexp2ps 越界输入。在 x86 上不明显，只是因为 x86 的 expf 处理下溢干净，且垃圾内容不同。
2. **向量化时对 "越界读" 要保持警惕**：`_mm512_load_ps` 一次读 16 个元素，`size` 不是 16 的倍数（尤其 `< 16`）时会读到缓冲区垃圾。任何向量化归约都要只在有效元素上计算。
3. **硬件快速数学指令有输入范围限制**：KNC `vexp2ps` 对结果低于次正规下界的输入返回 NaN 而非 0。需要处理下溢/溢出时，应钳位输入到安全区间（对 softmax/sigmoid 这类归一化操作，钳位不损失精度）。
4. 该修复对所有模型有益：softmax max_val bug 是潜在的通用缺陷，只是此前未被触发。

## 7. 相关文件

| 文件 | 说明 |
|---|---|
| `src/run.c` | 修复位置：MIC softmax（max_val + exp 钳位）、FFN sigmoid 钳位、k_bias 恢复 |
| `docs/QWEN_CN_STATUS.md` | Qwen 适配状态（已更新，纠正旧结论） |
| `docs/DEBUG_LOG.md` 12.14 | softmax 向量化记录 + 后续修复说明 |
| `docs/xeon-phi-7120p-summary.md` | KNC 内建函数可用性验证（vexp2ps 等） |
