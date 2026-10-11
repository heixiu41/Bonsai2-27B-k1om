# Xeon Phi 7120P (k1om) 推理：最终交付状态

> ## ⚠️ 更正（本文件写于修复轮之前）
>
> **"实测有效带宽 49 GB/s / 预期 1–4 tok/s" 那一段是错的。** 重做基准后的实测：
> 推理内核的访问模式（`rows-1360`）在 244 线程下达到 **196.43 GB/s**（6.87 GB buffer，3 次取最好）。
> 每 token 访存只需 **36 毫秒**，而实际每 token **3456 毫秒** ⇒ **访存占 1%，计算占 99%**。
> 早期那个 3.32 GB/s 来自一个只读 1 GB 的基准（大概率命中缓存）+ 计时器分辨率不足。
>
> **另外，当前速度是 0.268–0.290 tok/s**（IMCI 内核修复 + 向量化解包之后），
> 且 `[threads]` 参数、mode 白名单、隐藏状态地标都已在修复轮加入 —— 见 `HANDOVER.md`。

## 结论

**Ternary Bonsai 2 27B 已在这张卡上原生运行，输出与官方实现精确一致。**

```
top-1 = 11751   official = 11751 (' Paris')   MATCH
rank of 11751 = 0 / 248320
```

运行命令（默认路径，无需任何额外宏）：

```powershell
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
& $icc -mmic -O2 -std=c11 -qopenmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE `
       main_k1om.c model.c gguf.c -o bonsai_k1om -lm
```

## 性能（全部实测，244 线程）

| 配置 | 正确性 | decode |
|---|---|---|
| 标量 + OpenMP | ✅ | 0.161 tok/s |
| IMCI 内核（修一半 bug） | ✅ | 0.188 tok/s |
| **IMCI + 向量化解包（当前默认）** | ✅ | **0.267 tok/s** |

单线程 matvec 加速 **4.98x**；244 线程下只有 1.66x，因为并行后**内存带宽成为新瓶颈**
（实测 244 线程顺序读 3.32 GB/s，122 线程 6.51 GB/s）。

## 本轮修掉的两个真 bug

### 1. IMCI 内核每个 block 只算了一半的权重

```
根因：每 block 只做 4 次 16-wide FMA（wf+0/16/32/48 = 64 个 float），
      而 block 有 128 个权重，wf+64/80/96/112 从未被读。
      循环 j += 128 正常推进 → 每行恰好一半权重被静默跳过 → 结果恒为一半。
```

**诊断教训**：我前面所有"收窄"测试都在验证**中间量**，所以抓不到：

| 我测的 | 结果 | 为何漏掉 |
|---|---|---|
| `reduce(load(wf+0..48))` 单块 | ✓ -16 | 只覆盖被读的那一半 |
| 循环次数 | ✓ 40 | 循环确实跑了 40 次 |
| FMA 累加 4 次 | ✓ 64 | FMA 本身正常 |
| "每 lane 应有 -1280" | 实测 -640 | **我把 -640 当成了期望值** |

**规则**：断言「**结果 == 参考**」，而不是「某个中间量 == 我假设的值」。
还应检查 FMA 网络的**权重覆盖总数是否等于 QK_PQ2_0（128）**。

### 2. 向量化解包：KNC 上"头文件有声明" ≠ "能链接"

关键前提：**必须加 `-D__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES`**，
否则大部分链接失败会**伪装成编译错误**（`a value of type int cannot be used...` = 隐式声明）。

```
可链接：permutevar_epi32, swizzle_epi32, shuffle_epi32, permute4f128_epi32,
        slli/srli/srai_epi32, sllv/srlv_epi32, and/or/xor_si512,
        add/sub/mullo/abs/min/max_epi32, load/store_epi32,
        extload_epi32 (NONE+UINT8), extloadunpacklo/hi_epi32 (NONE+UINT8),
        alignr_epi32, set1_epi8/16, set_epi32/64, setzero_si512

运行时没有：_mm512_cvtepi32_ps (→ 不存在的 _mm512_cvt_roundepi32_ps),
            整个 unpacklo/hi_epi8/16/32/64 家族, shuffle_epi8,
            slli/srli_epi16, mullo_epi16, madd_epi16, maddubs_epi16,
            permutexvar_epi32, 以及所有 *loadu*/*storeu* 形式
```

**替代方案**：用 `_mm512_permutevar_epi32`（**索引在前**，全交叉 16 项 dword gather）
配一张 16 项 float LUT，一举替代缺失的 int→float 转换。

**差点静默埋进去的坑**：`extloadunpacklo(p, UINT8)` 会**静默补零** ——
越过 `p` 所在 64 字节对齐块之后的内容全变 0，不报错、不崩溃。
因为 `qs = blk+2` 且 stride 34 字节，`qs&63 > 32` 的 block 约占一半，
于是每 128 个权重里约 10 个被清零。正确做法是 `lo(p) + hi(p+64)`。
**只有逐元素比对能发现它。**

### 精度验证

```
4000 随机 block × 全部 64 种源对齐 = 32,768,000 次比较    MAX |diff| = 0
整个 10240×5120 矩阵 = 52,428,800 个权重                  MAX |diff| = 0
```

## 遗留问题与约束

| 项 | 说明 |
|---|---|
| **速度仍慢** | 0.267 tok/s。IMCI 单线程有 4.98x 余量，但并行后受内存带宽限制 |
| `lo/hi` 读越界 | 会读到 block 之后约 96 字节，**调用方需权重缓冲区后有可读余量**。若 `mmap` 到映射末尾，最后一个 block 要特殊处理 |
| COI 未完成 | 宿主侧已能用 Python `ctypes` 驱动（`COIEngineGetHandle` 成功），但缓冲映射未做。**若完成可省掉模型上传、腾出 6.8 GB 内存** |
| 无 C 分词器 | `main_k1om.c` 只吃 token id 数组 |
| `BONSAI_FORCE_SCALAR_PQ2_0` | 退路开关，回到已知正确的标量基线，用于二分数值问题 |

## 相关文档

- `docs/K1OM_BUILD_RECIPE.md` —— 交叉编译与部署配方（含 5 个坑）
- `docs/pq2_0_imci_decoder.md` —— 向量化解包完整报告
- `docs/KNC_INTEGER_SIMD.md` —— KNC 指令可用性实测（链接器为准）
- `docs/COI_ROUTE.md` —— COI 路线与 Python ctypes 驱动
- `docs/K1OM_TRANSFER_CHANNELS.md` —— SCIF vs SSH 通道实测
- `docs/PQ2_VECTORIZED_UNPACK.md` —— 中途记录（含后来被推翻的假设）
