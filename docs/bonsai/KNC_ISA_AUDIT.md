# KNC (k1om / 7120P) ISA 审计 — 2026-10-11

方法：icc 2017 的 `zmmintrin.h` 全集（2111 个唯一 `_mm512_*` 入口）→ 出货代码实际使用集 →
候选指令编译探针 → **-S 反汇编判定真硬件**（关键一步：`icc -mmic` 会把不存在的指令
静默编译成辅助函数调用，"编译通过"≠"硬件支持"）。

## 结论

1. **出货代码使用约 30 条真实硬件指令**（src 顶层 71 个名字里含注释与死代码）。
2. **编译器模拟名单**（编译通过但反汇编是 `call _mm512_xxx`，慢几个数量级，禁用）：
   - `_mm512_slli_epi16` / `_mm512_mullo_epi16` → `call`（坐实"KNC 无 16 位移位/乘法"）
   - `_mm512_loadu_ps` → **`call`（不是内联 loadunpack！）** 任何误入都是灾难
   - `_mm512_shuffle_epi8` / `_mm512_maddubs_epi16` / `_mm512_permutexvar_epi8` /
     `_mm512_permutex2var_epi32` / `_mm512_cvtph_ps` / `_mm512_exp2a23_ps`（KNL 名）/
     `_mm512_fixupnan_ps` / `_mm512_multishift_epi64_epi8` → 编译直接拒绝或模拟
3. **验证为真硬件的 KNC 特有指令**（反汇编单条助记符）：
   - `vrcp23ps` / `vrsqrt23ps` — 23-bit 精度倒数/倒数平方根（1 条指令）
   - `vexp223ps` — **2^(v / 2^24)，v 是 Q24 定点整数**（1 条指令，需配 `vcvtfxpntps2dq $80`）。
     用法：`vexp223ps(cvtfxpnt_q24(x·log2e))`。**更正**：早期这里写的是
     "2^x − 1，exp(x) = vexp223ps(x·log2e)+1"，2026-10-12 的探针实测推翻了它——
     那个 `+1` 会让 `exp(0)=2.0`；整数锚点（`v=0→1.0`、`v=±2^23→2^±0.5`）证明输入是
     2^24 定标而不是 float，输出不含 −1。精度实测最差相对误差 9.4e-7，
     深负数干净下溢到 0（不是 NaN）。详见 `ENGINE_TWEAKS_AB.md` §3。
   - `vscaleps` — a × 2^floor(b)（exp/ldexp 构件）
   - `vloadunpacklps/hps` — 真正的非对齐加载对（目前数据全对齐，用不上）
   - **免费 swizzle**：`vmovaps %zmm0{cdab}, %zmm0` — 寄存器移动/访存操作数可直接
     携带 {cdab} 等 swizzle 后缀，零额外成本。注意 swizzle 只在 128-bit 组内置换，
     不能跨半区（rmsnorm 的高半区提取仍需 permutevar）。
4. **F3 块循环（30 ops/128 权重）无指令级压缩空间**：唯一可能的大赢家是 16 位
   SIMD 操作（直接对 4-bit nibble 操作），已确认硬件不存在、编译器走模拟；
   gather 已实测慢；permutevar LUT 已是该 ISA 下的最优解码形态。

## 剩余可利用项（价值排序）

| 项 | 机制 | 预估 | 风险 |
|---|---|---|---|
| 向量 exp（vexp223ps+rcp23ps） | attention softmax / sigmoid gate / silu 的 expf 是 libm 标量 ~50cyc/个；向量版 16-wide ~几条指令 | ~0.3-0.5 ms/token（这些路径已并行化，基数小） | **已实测为负结果**：icc 早就把逐元素 `expf` 变成 `__svml_expf16`（`model.c` 汇编可见），手写 `vexp223ps` 实测只有 1.03x。不要再查这条 —— 见 `ENGINE_TWEAKS_AB.md` §3 |
| rsqrt23 进 rmsnorm | 129 次/token 的 1/sqrtf 是标量 | 噪声级 | — |
| loadunpack | 解除对齐约束 | 防御性，非性能 | — |

判定：vexp223ps 向量 exp 是唯一值得做 A/B 探针的指令级项，预期收益 <0.5 ms/token。
内核主循环（~190ms/token）在当前 ISA 下已到指令数下限。

## 数据存档

- 全集：`zmmintrin.h` 2111 个入口（大量 mask/maskz 变体 + SVML 数学库函数 + KNL/AVX512
  污染条目，**不能直接当硬件清单用**）
- 探针脚本模式见 src/probes/pq2isa_scan.ps1；本轮探针在 %TEMP%\isa_*.c / isa2_*.s
- 权威文档（未能在线获取，重定向循环）：Intel《Xeon Phi Coprocessor ISA Reference
  Manual》文档号 327364

## 官方 Intrinsics Guide 交叉核对（v3.6.9, 2024-07-12）

离线包（https://cdrdv2.intel.com/v1/dl/getContent/671338，data.js 全库 7146 条）已下载
解析：**当前官方指南已删除全部 KNC/IMCI 条目**——VEXP223PS / VRCP23PS / VRSQRT23PS /
VLOADUNPACK / VPMADD233PS / VSCALEPS / VPACKSTORE 全部 0 命中，连 KNL 的 AVX512ER
（rcp28/exp2a23/rsqrt28）也已移除。KNC 是 2013 年产品，EOL 后官方文档被清理。

**证据等级（本项目最终口径）**：
1. 卡上实测 + icc -S 反汇编探针（最权威，本轮已做）
2. icc 2017 zmmintrin.h（与 KNC 同年代，但含 AVX512/SVML 污染 + 静默模拟陷阱）
3. 官方在线文档（已不含 KNC，不可用）
