# V12 集成方案（下一步执行）

> ## ✅ 已完成 + 一处更正（后续轮次）
>
> **V12 已集成**（`src/pq2_0.h` 的 `pq2_0_matvec_knc` 现为 V12 解码器 + 每次调用
> 构建 xp），卡上端到端 1.645 tok/s。此后又做了一轮（小权重缓存 + Hadamard 去重 +
> GDN glue 并行化 + fwht 向量化），decode 达 **2.686 tok/s**，top-1 仍 11751 MATCH。
>
> **更正：本文下面"已排除的路"引用的"4.3x 差距"不存在。** 模型内 matvec 的
> "25.9 G w/s" 是 GB/s（打包字节）误标成 G 权重/s；真实值 25.9 × 128/34 =
> **97.5 G w/s**，与独立基准在真实 shape 混合下的 99.06 G w/s 一致
> （pq2call 实测）。内核在模型里一直跑在满速，没有可回收的 4.3x。
>
> ---
>
> 以下是原始计划（历史记录）。

> 这份说明让 V12 的集成可以在一个干净的会话里一次做完并验证。
> 我（上一轮）没有做，是因为上下文预算用尽，而集成**必须**配完整的验收流程。

---

## V12 是什么（已验证）

`src/bonsai/pq2v16.c` 里有 `v12_helper` / `v12_inline`。核心是 `pq2v15.c` 的 `blk()`：

```c
static inline void blk(const uint8_t *blkp, const float *xp0, float d,
                       const __m512 la0, const __m512 lb0, __m512 s[4]) {
    const __m512 vd = _mm512_set1_ps(d);
    const __m512i c0  = PQ2_IMCI_LOAD16(blkp + 2);
    const __m512i c1  = PQ2_IMCI_LOAD16(blkp + 18);
    const __m512i c0h = _mm512_srli_epi32(c0, 4);   /* 一次移位取字段 2/3 */
    const __m512i c1h = _mm512_srli_epi32(c1, 4);
    const __m512 la = _mm512_mul_ps(la0, vd), lb = _mm512_mul_ps(lb0, vd);
    const __m512i lai = _mm512_castps_si512(la), lbi = _mm512_castps_si512(lb);
    g[0] = cast(permutevar(c0,  lai));   g[1] = cast(permutevar(c0,  lbi));
    g[2] = cast(permutevar(c0h, lai));   g[3] = cast(permutevar(c0h, lbi));
    g[4] = cast(permutevar(c1,  lai));   g[5] = cast(permutevar(c1,  lbi));
    g[6] = cast(permutevar(c1h, lai));   g[7] = cast(permutevar(c1h, lbi));
    /* 8 x FMA against xp0..xp0+127 */
}
```

两处减指令：
1. **缩放折进 LUT**：`la = la0*vd`、`lb = lb0*vd` ⇒ 每块 **2 次 vmulps**（原 8 次）
2. **两张 LUT**：`la0[j]=(j&3)-1`、`lb0[j]=((j>>2)&3)-1`，同一个未移位索引就给出字段 0 和 1
   ⇒ 移位只用于字段 2/3，**2 次 srli**（原 6-8 次）

指令数（反汇编实测）：**40 → 30/31 per 128 权重**，实测 **1.55-1.58x** 内核级。

验证：**26,832 项元素检查，0 失配**；多块相对差 0.000e+00（`pq2v12_card_evidence.txt`）。

---

## 集成的关键难点（必须先解决）

**V12 需要 `xp` —— 一个置换过的输入激活副本：**

```c
static void build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[((b*2 + h)*4 + f)*16 + l] = x[b*128 + 64*h + 4*l + f];
}
```

**⚠️ `mm_folded` 只被调用一次，但 `pq2_0_matvec_knc` 内部是 `#pragma omp parallel for`（244 线程）。
在 kernel 内部构建 `xp` 会有 244 个线程争写同一个缓冲区 —— 数据竞争。**

**必须做在 `mm_folded` 里（并行区之外），构建一次、传进 kernel。**

### 推荐做法：给 kernel 加一个 `xp` 参数

```c
/* pq2_0.h —— 两处都要改 */
static inline void pq2_0_matvec_knc(const float *xp, float *y,
                                    const uint8_t *w, int n, int d);
static inline void pq2_0_matvec_ref(float *y, const float *x,
                                    const uint8_t *w, int n, int d);  /* 不变 */

#ifdef __MIC__
    pq2_0_matvec_knc(m->sc.fold_xp, y, w, n, d);      /* V12 路径 */
#else
    pq2_0_matvec_ref(y, xin, w, n, d);                /* 宿主参考，不变 */
#endif
```

在 `model.c` 的 `mm_folded` 里（**`#ifdef __MIC__` 内**）：

```c
#ifdef __MIC__
    /* V12 需要置换过的输入；在并行区之外构建一次。
     * xp 是 scratch 里新增的 17408 float（64 字节对齐），约 70 KB。 */
    build_xp(xin, m->sc.fold_xp, n);
    pq2_0_matvec_knc(m->sc.fold_xp, y, w, n, d);
#else
    pq2_0_matvec_ref(y, xin, w, n, d);
#endif
```

`model.h` 的 scratch 结构里加 `float *fold_xp;`，
`model.c` 的分配处用 `aligned_f32(17408, 64)` 分配、在 `bonsai_model_close` 里释放。

**`n` 必须是 128 的倍数** —— 本模型全部满足（5120/6144/17408，已核验 402 个张量），
但要在调用点加断言或回退到 `pq2_0_matvec_ref`。

---

## 验收标准（不可跳过）

```
1. 宿主编译 + k1om 编译   exit 0
2. 宿主回归               ALL CHECKS PASSED（5 阶段 + 4 个隐藏状态地标）
                          top-1 = 11751 MATCH、l_out-63 OK
   ⚠️ V12 改变了 FMA 配对顺序 ⇒ 会有 ~1e-7 量级的求和顺序差异，这是预期的
3. 卡上                    top-1 = 11751 MATCH、rank 0
4. 性能                    预期内核 1.55x、端到端约 1.17x（1.645 → 1.93 tok/s）
5. 逐元素验证             用 pq2cost.c 的 verify_vs 框架（64 对齐 × 128 单位向量 + 行和）
```

**第 2 步不能只看 top-1** —— 本项目已实测：RoPE 配对错误时 top-1 仍然 MATCH。

---

## 已经排除的路（不要再试，全部有实测）

```
❌ 工作集/缓存        pq2ws: 6 GB vs 23 MB 只掉 3.6%
❌ parallel region    pq2reg: 401 次 x 34 us = 13.6 ms/token，占 matvec 5%
❌ 线程闲置/不均      pq2util: 244/244 线程报告，imbalance 1.02x
❌ 更便宜的指令        pq2shuf/pq2pipe: 全部 512 位指令同价，单流水线
❌ 预解包             内存装不下（int8 需 26.9 GB > 15.5 GB）
❌ 批量 decode        pq2batch: B=16 只 1.11x（FMA 随 B 线性增长）
❌ 批量 prefill       pq2pf: 每 token 成本恒定；且带宽不是瓶颈（利用率 15%）
❌ 位平面 + popcount  无向量 popcount；且 x 是 fp32，乘法省不掉
❌ GEMM（batch=1）    无复用，本就是 matvec
```

**唯一剩下的杠杆就是 V12/F3 的减指令数。**
