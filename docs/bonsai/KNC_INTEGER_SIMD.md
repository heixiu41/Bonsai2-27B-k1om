# k1om 指令集：以链接器为准的可用性清单

> ## ⚠️ 探测方法本身有个陷阱（我踩了两次）
>
> **`undefined reference` 只在调用真的到达链接器时才出现。**
> 若内建函数的结果没有被使用，编译器会把整个调用**当死代码消除**，
> 于是链接"成功" —— **给出假阳性**。
>
> 我第一次探测（`intr_probe.c`）把结果存进一个数组但**没有打印**，
> 于是对**每个宏单独编译**（其他宏关闭、程序几乎全空）时，
> 编译器消掉了那些存储 → 5 个内建函数被误判为"可链接"：
> `cvtepi32_ps`、`slli_epi16`、`srli_epi16`、`srai_epi16`、`mullo_epi16`。
>
> **后来用"把结果打印出来"的程序（`pq2settle.c`）复核，全部链接失败**，
> 错误信息与我最初的结论完全一致。
>
> **正确的探测写法**：让结果**可观察**（打印 / 写入 volatile / 影响返回值），
> 或者把全部候选放在同一个程序里一起编译（一个失败就暴露，但看不出是哪个）。
> **最稳的是两者结合**：单宏编译 + 强制可观察。
>
> **教训**：这和项目里其他几次错误同类 —— "看起来成功"不等于"真的成功"。

## 必需的前置宏

```
-D__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES
```

**没有它，大部分链接失败会伪装成编译错误**：
```
error: a value of type "int" cannot be assigned to an entity of type "__m512i"
```
这其实是**隐式声明**（函数未声明 → 编译器假定返回 int），不是类型问题。
我就是被这个误导过，以为是写出法错了。

---

## 能链接（实测，且结果经数值核对）

```
向量整数（32 位）
  _mm512_permutevar_epi32        ← 索引在前；全交叉 16 项 dword gather；vpermd
  _mm512_mask_permutevar_epi32
  _mm512_permute4f128_epi32
  _mm512_swizzle_epi32           _mm512_shuffle_epi32
  _mm512_slli_epi32 / _mm512_srli_epi32 / _mm512_srai_epi32
  _mm512_sllv_epi32 / _mm512_srlv_epi32
  _mm512_and_si512 / _mm512_or_si512 / _mm512_xor_si512 / _mm512_andnot_si512
  _mm512_add_epi32 / _mm512_sub_epi32 / _mm512_mullo_epi32
  _mm512_abs_epi32 / _mm512_min_epi32 / _mm512_max_epi32
  _mm512_mask_blend_epi32 / _mm512_mask_mov_epi32 / _mm512_mask_set1_epi32
  _mm512_cmpeq_epi32_mask / _mm512_cmplt_epi32_mask
  _mm512_alignr_epi32
  _mm512_set1_epi8 / _mm512_set1_epi16
  _mm512_set_epi32 / _mm512_set_epi64 / _mm512_setzero_si512

加载/存储
  _mm512_load_epi32 / _mm512_mask_load_epi32 / _mm512_store_epi32
  _mm512_extload_epi32                       (NONE 与 UINT8 两种模式)
  _mm512_mask_extload_epi32
  _mm512_extloadunpacklo_epi32 / _mm512_extloadunpackhi_epi32
  _mm512_mask_extloadunpacklo_epi32
  _mm512_i32extgather_epi32

字节/字级：**没有任何可用的 512 位字节/字 ALU 指令。**

> 我曾在本文件里写「`_mm512_add_epi8` / `sad_epu8` / `max_epu8` 可用（来自 dvec.h）」
> —— **那条是错的。** 用正确写法复核（结果可观察 + 数值核对）：
> ```
> _mm512_add_epi8    undefined reference
> _mm512_sad_epu8    undefined reference
> _mm512_max_epu8    undefined reference
> ```
> 我混淆了 `dvec.h` 里 **128 位 `_mm_` 形式**与 **512 位 `_mm512_` 形式**。
> 而且 128 位的 SSE 形式也被编译器拒绝（`Opcode unsupported: pand`）。
> ⇒ **把 KNC 的整数单元当作「只有 32 位元素」。**
>
> 这条排除了一整类思路：**用字节/字 SWAR 做 2-bit 字段提取**。

浮点
  _mm512_load_ps / _mm512_store_ps / _mm512_setzero_ps / _mm512_set1_ps
  _mm512_add_ps / _mm512_sub_ps / _mm512_mul_ps / _mm512_fmadd_ps
  _mm512_reduce_add_ps / _mm512_castsi512_ps
```

## **不能链接**（运行时库没有该符号）

```
_mm512_cvtepi32_ps        → 被降低为 _mm512_cvt_roundepi32_ps，该符号不存在
整个 unpacklo/hi_epi8/16/32/64 家族
_mm512_shuffle_epi8
_mm512_slli_epi16 / _mm512_srli_epi16 / _mm512_srai_epi16     ← 复核确认
_mm512_mullo_epi16 / _mm512_madd_epi16 / _mm512_maddubs_epi16
_mm512_mulhi_epi16
_mm512_permutexvar_epi32 / _mm512_permutex2var_epi32 / _mm512_permutexvar_epi8
_mm512_permutevar_ps
_mm512_sub_epi8
_mm512_bslli_epi128 / _mm512_bsrli_epi128
_mm512_alignr_epi8 / _mm512_alignr_epi64
_mm512_shuffle_i32x4
所有 *loadu* / *storeu* 形式   ← k1om 没有非对齐普通加载
```

## **编译器拒绝**

```
_mm512_cvtepu8_epi32    catastrophic error: unsupported data types or intrinsics on target
_mm512_cvtepi8_epi32
_mm512_cvtepu16_epi32
_mm512_cvtepu8_epi16
_mm512_cvtepi16_epi32
```

## 头文件里声明很多，但**声明 ≠ 可用**

`zmmintrin.h`（595 KB）声明了 `_mm512_shuffle_epi8`、`_mm512_permutexvar_epi32`、
`_mm512_madd_epi16`、`_mm512_bslli_epi128`、`_mm512_broadcastb_epi8`
等一大堆 —— **链接器全部拒绝**。
`_mm512_broadcastb_epi8` 甚至不通过编译（参数类型不符，那是 KNL 的签名）。

**所以：想用某个指令，先让链接器判定，再让数值核对确认。**

---

## 这对算法设计意味着什么

### KNC 没有低精度整数乘加

不能做"整数乘加 + 最后转 float"的定点累加，因为**缺的正是成对乘加**：
- 没有 `madd_epi16`（16 位成对乘加 → 32 位）
- 没有 `maddubs_epi16`（字节×字节 → 16 位）
- 没有 `dpbusd`/`dpwssd`（VNNI，KNL 2016 才有）

这是 KNC（2012）与 KNL 的关键差异。

### int→float 转换的替代方案

既然 `_mm512_cvtepi32_ps` 不可用，而三值码只有 4 个值 `{-1, 0, +1, +2}`，
**用 16 项 LUT + `_mm512_permutevar_epi32` 一次置换即可**，不需要数值转换。

实测对比：
```
16 项 LUT + permutevar     1254.4 M 权重/s   ← 采用这个
IEEE-754 bias trick         290 M 权重/s    ← 慢 2.2 倍
```
bias trick（`bitcast_ps(c + 0x4B000000) - 8388608.0f`）**数学上对 `|c| ≤ 2` 精确**，
但把一次 `vpermd` 换成了 `vpandd + vpaddd + vsubps`，
而 **KNC 的 512 位整数 ALU 是慢路径**。

> 注：我自己试 bias trick 时**负值出错**（`-1` 变 `-0`），而子代理复现时是精确的。
> 差异未查清 —— 可能是我的版本上游某处已经错了。若要用它，先做最小重现。

---

## 对齐要求（实测，会 SIGSEGV）

| 指令 | 要求 | 违反后果 |
|---|---|---|
| `_mm512_load_ps` / `_mm512_load_epi32` | **64 字节** | SIGSEGV |
| `_mm512_extload_epi32` NONE | 64 字节 | SIGSEGV |
| `_mm512_extload_epi32` UINT8 | 16 字节 | — |
| `_mm512_extloadunpacklo_epi32` NONE | 4 字节（偏移 1/2/3 会 SIGSEGV） | — |
| `_mm512_extloadunpacklo_epi32` UINT8 | **任意字节偏移**，但有陷阱 ↓ | 见下 |

### `extloadunpacklo` UINT8 的静默陷阱

它会**零填充**越过 `p` 所在 64 字节对齐块的一切内容 —— **不报错、不崩溃**：
```
p = base+52:
  只用 lo                   → 109 116 123 130 137 144 179 186 0 0 0 0   ← 静默补零
  lo(p) + hi(p+64)          → ... 193 200 207 214                       ← 正确
```

因为 `qs = blk+2` 且 stride 是 34 字节，`qs&63 > 32` 的块约**占一半**，
于是每 128 个权重里约 10 个被静默清零。
**正确写法是 `lo(p) + hi(p+64)`。只有逐元素比对能发现这个 bug。**

### 越界读足迹（实测）

```
足迹 = [blk, blk+81]，即最多越过 34 字节块末尾 48 字节，且最多向前 61 字节
与源对齐无关（64 种字节相位同阈值）
纯读：毒化那 48 字节（0x00 vs 0xA5）不改变任何解码结果（0/128）
```
`gguf.c` 因此分配 `file_size + 64` 并清零尾部；**移除这个余量前先看 `gguf.h` 的说明**。

---

## 探测方法

`src/bonsai/intr_probe.c` 用 `#ifdef TRY_*` 开关逐个打开候选指令，
**让链接器决定**。构建/上传/运行脚本：
```
tools/probe_links.ps1      （需先 Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass）
tools/mic.ps1              （通用的构建+上传+运行）
```

## 相关文件

- `src/bonsai/pq2_0.h` —— 向量化解码器（用的是 LUT + permutevar 方案）
- `docs/pq2_0_imci_decoder.md` —— 解码器完整报告（含 0 失配证据）
- `docs/PQ2_VECTORIZED_UNPACK.md` —— ⚠️ 中途记录，含已被推翻的假设
