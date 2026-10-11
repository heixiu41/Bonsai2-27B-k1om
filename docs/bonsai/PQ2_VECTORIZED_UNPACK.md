# 向量化解包（PQ2_0）—— 进展、实测数据与卡点

> ## ⚠️ 本文件是**中途记录**，其中两条结论已被推翻
>
> **1. "IMCI 缺陷依赖具体数据"是错的。** 那条结论来自 `pq2vec.c` 的 `half_test()`，
> 而它自身有**堆越界读写**（只分配 1 行 `row_bytes` 却按 `d` 行索引 `w + i*row_bytes`），
> 那些 -135631 / -373223 是垃圾内存。真实原因见下条。
>
> **2. IMCI "恰好一半"的根因是 FMA 覆盖不足，已于修复轮解决。**
> 内核每个 block 只做 4 次 16-wide FMA（`wf+0/16/32/48` = 64 个 float），
> 而 block 有 128 个权重 —— `wf+64/80/96/112` **从未被读**，
> 循环 `j += 128` 又正常推进，于是每行恰好一半权重被静默跳过。
> 修复：每个累加器再加一次 16-wide FMA，覆盖全部 128 个权重。
> 已逐元素验证：12/12 形状与标量参考一致，最大相对误差 2.1e-7~8.8e-7。
>
> **3. 越界读是 48 字节不是 96**（守卫页实测，见 `pq2_0_imci_decoder.md` 顶部更正）。
>
> **最终的向量化解码器**（由另一代理完成）见 `docs/pq2_0_imci_decoder.md`：
> 用 16 项 LUT + `_mm512_permutevar_epi32` 替代缺失的 int→float 转换，
> 5.56x（DRAM 流式）/ 11.13x（L2 驻留），705,536 次元素比较 **0 失配**。
>
> 下面保留当时的原文，仅作过程记录 —— **不要据此下结论**。

## 为什么做这个

实测证据表明瓶颈**不是内存带宽**：

```
推理内核实际吞吐              1.36 GB/s
单线程顺序读上限（实测）       1.57 GB/s
244 线程顺序读                3.32 GB/s
122 线程顺序读                6.51 GB/s
```

标量路径跑 244 线程却只达到**单线程**的访存速度 —— 说明循环体（解包）是瓶颈，
不是内存。所以向量化解包是唯一能带来数量级改善的改动。

## KNC 可用的解包指令（查 `dvec.h` / `micvec.h` 确认）

```
_mm512_slli_epi32 / _mm512_srli_epi32 / _mm512_and_si512
_mm512_unpacklo_epi8 / _mm512_unpackhi_epi8   （以及 epi16 / epi32）
_mm512_mullo_epi32   _mm512_add_epi32
_mm512_load_epi32    _mm512_store_ps
_mm512_sad_epu8      _mm512_avg_epu8
```

**没有**：`_mm512_alignr_*`、`_mm512_permutexvar_*`、`_mm512_dpbusd`（VNNI，KNL 才有）、
`_mm512_maddubs_epi16`、`_mm512_madd_epi16`。

## 已实现且**正确**的部分

`src/bonsai/pq2vec.c::unpack_codes()` —— 位提取实测与标量完全一致：

```
codes[0..7]  = 2 -1 -1 -1 1 1 -1 -1
expect       = 2 -1 -1 -1 1 1 -1 -1      ✓ 完全一致
raw byte qs[0]=0x03
```

**但当前版本每个 code 仍是标量取出**（`out[idx] = ((byte >> (k*2)) & 3) - 1`），
所以只有 1.15x 加速。**真正的位提取向量化还没做。**

## 卡点：int32 → float 转换

### `_mm512_cvtepi32_ps` 在 KNC 上链接失败

```
undefined reference to `_mm512_cvt_roundepi32_ps'
```

它被降低到一个 KNC 上不存在的 round 版本。

### 改用 IEEE-754 "bias trick"（对 |c| ≤ 2 精确）

```c
// (float)c == bitcast_ps(c + 0x4B000000) - 8388608.0f
const __m512i bias_i = _mm512_set1_epi32(0x4B000000);
const __m512  bias_f = _mm512_set1_ps(8388608.0f);
const __m512i b = _mm512_add_epi32(c, bias_i);
const __m512  f = _mm512_sub_ps(_mm512_castsi512_ps(b), bias_f);
```

### **实测有 bug**：负值错

```
scalar[0..5] = 2 -1 -1 -1 1 1
vector[0..5] = 2 -0 -0 -0 1 1     ← 只有 -1 错，+2/+1 都对
```

中间值 `codes[]` 是对的，所以问题在**转换本身或 `_mm512_castsi512_ps` 的行为**。

**下一步**：打印 `_mm512_add_epi32(c, bias_i)` 的原始位模式确认。

## 备选方案（若 bias trick 走不通）

1. **查表法** —— `_mm512_swizzle_epi32` / `_mm512_permute4f128_epi32` 把 code 映射到预置 float
2. **改为定点累加** —— 全程用 `_mm512_add_epi32` 累加整数，最后一次性转 float。
   对三值码这最自然，且**完全避开浮点转换**
3. `_mm512_cvtfxpnt_round_adjustepi32_ps`（`micvec.h` 里有，但需设舍入模式）

## 顺带推翻的一个假设

曾怀疑 IMCI 点积"恰好一半"的缺陷源于 **`wf` 栈数组对齐**（`fold_x` 已经从 `calloc`
改成 64 字节对齐的 `aligned_f32`）。**实测否定**：

```
正确值         = -40960
heap staging   = -135631    ✗
stack staging  = -373223    ✗
```

**两者都错，且错得不一样** —— 不是对齐问题。而且这次的数值与更早的 `pq2mini`
测试（那里 uniform 权重下是"恰好一半"）不同，说明该缺陷**依赖具体数据**，
或者存在内存问题。需要重新用受控输入测。

## 相关文件

- `src/bonsai/pq2vec.c` —— 本轮的测试程序（解包正确性 + 吞吐 + stack/heap 对比）
- `src/bonsai/pq2mini.c` / `pq2narrow.c` / `pq2diag.c` —— 更早的 IMCI 缺陷诊断
- `src/bonsai/pq2_0.h` —— IMCI 内核（当前**默认禁用**，走标量+OpenMP）
- `src/bonsai/bwbench.c` —— 带宽微基准
