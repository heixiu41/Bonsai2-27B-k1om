# 代码审查结果汇总（三轮独立审查）

审查对象：`src/bonsai/` 下的 k1om 推理引擎。三个互不重叠的审查代理并行工作。

> **ℹ️ 本文档中出现的带宽数字（3.32 / 6.51 GB/s）是"我最初的错误测量"的引用，
> 用来解释我为什么会得出错误结论 —— 它们不是当时的真实带宽。**
> 重做基准后的实测（244 线程，6.87 GB buffer，3 次取最好）：
> `rows-1360` **196.43 GB/s**、`contiguous` 62.45、`rows-340` 103.54、`interleaved` 47.37。
> 访存只占每 token 时间的 **1%**。详见 `PERFORMANCE_OPTIMIZATION.md` 顶部的更正说明。

> **⚠️ 旧副本已删除**：仓库根目录曾有一套更短的陈旧副本
> （`pq2_0.h` 228 行 —— **只含 4 次 fmadd，即那个"恰好一半"的 bug**，
> 另有 `model.c`/`model.h`/`gguf.c`/`gguf.h`/`test_forward.c`）。
> 实测：从根目录执行 `icc -mmic ... src/bonsai/main_k1om.c model.c gguf.c`
> **会链接成功**（exit 0），产出一个前向是半权重内核的二进制 —— **静默地**。
> 这六个文件已删除（无任何脚本引用它们）。权威版本只在 `src\bonsai\`。

---

## 第三轮审查（独立复现，与第二轮不重叠）

### ⚠️ F2 —— **"oracle" 实现的 RoPE 与引擎不一致，而 top-1 检查抓不到**

`scripts/bonsai_forward.py:61-78` 的 `rope_multi()` 用 `x[:64].reshape(nr,2)`，
即**相邻配对** `(2i, 2i+1)` —— 而它**自己的文档字符串**（第 66 行）却写
"neox-style rotate_half pairing"，`model.c:193-208` 用的是**前后半配对** `(i, i+32)`，
从 fork 源码提取的 `BONSAI2_FORWARD_SPEC.md:375` 也明确写"**不是相邻配对**"。

**A/B 实验**（只改配对方式，宿主构建）：
```
前后半配对（model.c）  → l_out-63 first3_err 4.60e-02 OK, ‖x‖ 比值 0.9997, ALL CHECKS PASSED
相邻配对（oracle）     → first3_err 1.01e-01 MISMATCH, 比值 1.0026
```
**所以 model.c 是对的，oracle 是错的。**

**最危险的一点**：两种配对都给出 `top-1 = 11751`、`rank 0`。
**整个项目的正确性判据（rank-0/top-1）检测不出 RoPE 配对错误。**
只有隐藏状态检查（`l_out-63`）能抓到。

> **行动项**：在使用 oracle 去"修正" `model.c` 之前必须先修 oracle；
> 并在正确性门槛里**必须**保留隐藏状态检查，不能只靠 top-1。

### F1 —— 解码器越界量实测 **48 字节**（文档写"约 96"，偏保守）

用 `PROT_NONE` 守卫页在卡上实测：尾部余量 0..47 字节时全部触发故障，**首个安全值 = 48**；
**全部 64 种源对齐给出相同阈值**，所以足迹是 `[blk, blk+81]`，与对齐无关。
把那 48 字节填 `0x00` 或 `0xA5` **不改变任何解码结果**（0/128），
所以是纯粹的越界**读** → SIGSEGV 风险，**不会静默产生错误数值**。

**今天不崩的原因**（两个幸运因素叠加）：
1. 张量表恰好以 F32 张量 `blk.63.post_attention_norm.weight` 结尾，
   最后的 PQ2_0 张量 `blk.63.ffn_up.weight` 距 EOF 还有 20,480 字节
2. `malloc(7206168928)` 的页对齐余量还有 1696 字节

**重新导出模型让任一 PQ2_0 张量排最后，或文件大小满足 `(size+16) % 4096 == 0`，
就会读到分配之外 → 运行时 SIGSEGV，构建期毫无提示。**

**✅ 已修**：`gguf.c` 改为 `malloc(file_size + 64)` 并把那 64 字节清零。

### F4 —— `x` 的 64 字节对齐是"靠运气满足"的前置条件

实测：`pq2_0_matvec_knc` 在 `x` 为 16 字节对齐时 **SIGSEGV**，4 KB 对齐正常。
引擎之所以满足，只因为**每个 `mm_folded` 调用都传了非 NULL 的 `signs`**，
从而把输入绕经 64 字节对齐的 `fold_x`。而 `mm_plain()` 会把调用者的
`calloc` 缓冲区直接送进内核 —— 它未被调用，但**留着**，且**没有宿主对应路径**，
所以这个 bug 只会在卡上出现。**建议删除 `mm_plain` 或让其绕经 `fold_x`。**

### F5 —— 非 PQ2_0 编码被接受后按 PQ2_0 解码

`GGML_TYPE_PTQ1_0`(143) 在 `gguf.h` 里被赋予与 PQ2_0 相同的 34 字节块大小；
实测能加载成功（`typeaccept.exe`：`SUCCESS, type=143, nbytes=136`）。
`dequant_to_f32` 会拒绝它，但 **`mm_folded`/`embed_row` 从不检查 `t->type`** →
用 PQ2_0 的码表解码 → **静默产出垃圾 logits，无任何报错**。

### F8 —— 一个"永远不会失败"的测试

`BONSAI_IMCI_PQ2_0` 是**幽灵宏**（`probe_i.c`、`pq2matvec.c` 依赖它）。
在宿主机上 `pq2_0_matvec_knc` 与 `pq2_0_matvec_ref` 是**同一个地址**，
所以这两个测试是在**拿参考实现和自己比**，永远通过。
（卡上两者不同，所以在卡上才有意义。）

---

## 一、已应用的修复

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `pq2_0.h:325, 391` | **tail block 寻址差一个 block**：行步长 `(n/128)*34`，tail 用 `(j/128)*34`，`j≥nn` 时两者相等 → 读到**下一行的第一个 block**；最后一行越界 | 改用向上取整步长 `rb = (n+127)/128*34` |
| 2 | `pq2_0.h:311` | 注释称内核"默认禁用"，但 `#ifndef` 实际**启用了**它 | 注释改为与代码一致 |
| 3 | `gguf.c` | ① 元数据值在边界检查**前**读取；② `selftest` 求和可回绕导致误报成功；③ 张量载荷从不校验是否超出 `file_size` | 三处均加边界/溢出检查 |
| 4 | `gguf.c:385` | 解码器越界读 48 字节（F1） | `malloc(file_size + 64)` + 清零 |
| 5 | 根目录 | 6 个陈旧副本，其中 `pq2_0.h` 含"一半"bug；混合编译命令会**静默产出错误二进制** | 已删除（无脚本引用） |

**验证**：宿主 `zig` 编译 exit 0；k1om `icc` 编译 exit 0；退路编译 exit 0；
宿主 `test_forward` **ALL CHECKS PASSED**（精确匹配 11751）；
k1om `test_gguf` **41/41 通过**。

---

## 二、审计纠正我的两个错误假设

### k1om 是 **LP64**，不是 32 位 `long`
我一直以为卡的 `long` 是 32 位，据此加了 `-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE`，
还当成"关键坑"写进文档。**实测否定**：在 mic0 上读 3 GB 文件的探针给出
`sizeof(long)=8`、`sizeof(off_t)=8`、`ftello` 返回 3145728000，
**去掉那两个宏行为完全一致**；sysroot 的 `bits/wordsize.h` 写着 `__WORDSIZE 64`。

### 最后张量紧贴 EOF，余量为 0
```
最后结束的: blk.63.post_attention_norm.weight (F32)  end = file_size  → 距 EOF 0 字节
PQ2_0 最后: blk.63.ffn_up.weight                     距 EOF 20,480    → 安全
```
我只在 PQ2_0 里找"最大风险"，漏了**整文件最后是个 F32 张量、零余量**。

### 带宽：我的测量错了 59 倍 ⚠️
```
第一次基准（1 GB buffer，可能命中缓存）:  244 线程 3.32 GB/s, 122 线程 6.51 GB/s
重做的基准（6.87 GB，3 次取最好）:
     contiguous    62.45 GB/s
     rows-1360    196.43 GB/s   ← 推理内核正是这个访问模式
     rows-340     103.54 GB/s
     interleaved   47.37 GB/s
```
**196 GB/s 是峰值的 56%。** 按此计算每 token 只需 `7.137/196 = 36 毫秒` 访存，
而实际每 token 3456 毫秒 → **访存占 1%，计算占 99%**。
我之前基于 3.32 GB/s 得出的"带宽会挡住优化"结论**完全错误**。

---

## 三、已报告但未修的发现

### 高危（畸形输入触发，正常模型不触发）

| # | 位置 | 问题 |
|---|---|---|
| F2' | `model.c:539` | `(size_t)(ck-2)` 下溢：`ssm_d_conv==1` 时算出 `(size_t)-1*conv_dim`，向缓冲区**之前** memcpy 40 KB。`calloc(0)` 返回非 NULL，检查通过 |
| F3' | `model.c:267, 479` | 先做除法再校验：`value_dim / ssm_dt_rank`、`rep = vh/kh` 在几何检查**之前**执行 → 实测 SIGFPE |

### 中危

| # | 位置 | 问题 |
|---|---|---|
| F4' | `model.c:170` | `mm_plain` 是对齐地雷（见上） |
| F5' | `model.c` 多处 | 数值路径**从不检查张量类型**（见上） |
| F6' | `model.c:75-100` 等 | `dequant_to_f32` 无容量参数；`hadamard_forward` 无 `n%block` 检查（`n%1024!=0` 会越界写最多 1023 个 float） |
| F7' | `model.c:362-367` | ~25 个分配只校验了 5 个；环境（16 GB 卡 + 7.2 GB 模型）下 OOM 是现实风险 |
| F5'' | `main_k1om.c:88-91` | 解码循环的 forward 失败被丢弃，退出码仍可为 0 |
| F6'' | `main_k1om.c:33` | `mode` 是黑名单；文档写的 `[threads]` 参数**从未被解析** |

### 低危 / 信息

| # | 位置 | 问题 |
|---|---|---|
| F7'' | `model.c` 14+ 处 | `dequant_to_f32`/`mm_folded` 返回值全被忽略；张量缺失时留下**陈旧 `y`**（logits 保留上一 token 值），forward 仍返回 0 |
| F8' | `pq2_0.h:203` | 越界读文档写"约 96 字节"，实测 **48** |
| F9' | `pq2_0.h:445` | `hadamard_forward` 的 `inverse` 参数**完全没作用**（两分支字节相同） |
| F10' | `pq2_0.h:90` vs `:325/391` vs 存储格式 | `n%128!=0` 有**三种约定**（ceil 填充 / 拒绝 / floor 打包），而实际上不可达 |
| F11' | `model.c:335` | `s->att` 分配+释放但从未使用；`bonsai_generate` 从未被调用 |
| F12' | `gguf.h:26-28` | 注释说文件"mmap 只读、从不拷贝"，实际是堆读取 —— 若有人"修正"加载器去匹配注释，**F1 就会变成活的 SIGSEGV** |

---

## 四、经实测确认**正确**的部分

| 项 | 证据 |
|---|---|
| 四个 `__MIC__` × `BONSAI_FORCE_SCALAR_PQ2_0` 组合 | `-E` 确认各自选择正确，全部编译通过（icc + zig） |
| 向量化解码器 | 卡上 **705,536 次元素比较**：全部 64 种对齐、`00/FF/55/AA/0F/F0/33/CC` 均匀模式、5000 个随机块（含 NaN/次正规/负 scale）—— **0 失配，逐位一致** |
| 8-FMA 权重覆盖 | `n ∈ {128..17408}` 与 `{5,127,129,130,136,192,200,255}`、`d ∈ {1,3,24}` 上 knc 与 ref 一致，最大相对 1.6e-5；毒化尾部余量不改变结果 |
| 对齐要求 | 真实且已满足：`wf` 必须 64 字节对齐（4 字节时 SIGSEGV），`x` 同（16 字节时 SIGSEGV）；`fold_x` 是唯一进入 `_mm512_load_ps` 的缓冲区 |
| `aligned_f32()` 簿记 | k1om 与宿主各 **27,000 次分配**：`p%64==0` 恒成立，`p-raw ∈ {16,32,48,64}`，`p-8` 始终在分配内 |
| `bonsai_fp16_to_fp32` | **全部 65,536 个 half 值**与 numpy 逐位一致（2046 个 NaN 按 NaN 比较） |
| 码表 | 32,768 次检查 0 失配 |
| Hadamard | 与显式奇偶矩阵一致到 4.8e-7，符号顺序正确 |
| `model.c` vs oracle 逐行 | GDN 的 decay/sk/d/状态更新顺序/qscale 等价、`h%kh` 头映射、`ssm_out` 置换、逐头 `[Q\|gate]` 切分、逐头 q/k RMSNorm、GQA、softmax、gate、embedding 的 H-后-符号 —— **全部一致。只有 RoPE 分歧（F2）** |
| 加载器（新版本） | 7 个畸形文件全部干净拒绝（旧版本 6 个里有 5 个越界读）；真实模型仍 `ALL CHECKS PASSED`，"regions tile the file exactly" |
| 并发编辑无回归 | 宿主 `test_forward` → ALL CHECKS PASSED、top-1 11751、rank 0；卡上 → MATCH、rank 0、0.267 tok/s |

---

## 五、建议的修复优先级

```
1. 【高】修 scripts/bonsai_forward.py 的 RoPE 配对（F2）
   —— 否则未来有人用它当 ground truth 去"修" model.c，会把对的改错
2. 【高】正确性门槛必须保留隐藏状态检查，不能只靠 top-1
   —— 实测 RoPE 配错时 top-1 仍然 "MATCH"
3. 【中】F2'/F3'（model.c 下溢 + 除零，各 1-2 行）
4. 【中】删除 mm_plain；数值路径加张量类型检查
5. 【中】所有分配检查 + 错误传播
6. 【低】文档更新：越界读是 48 而非 96；k1om 是 LP64；
           `inverse` 参数无效；`gguf.h` 的 mmap 注释与实现不符
```

## 六、未覆盖

- 前向数值正确性 vs Python oracle 的**完整**比对（只做了常量核对与逐行阅读）
- `src/run.c`/`runq.c`/`bfp16.c`/`win.c`（无关的 llama2.c 部分，约 190 KB）
- `scripts/` 下的 `export*.py`/`model.py`/`tokenizer.py`/`sample.py`/`chat_window.py`/`coi_probe.py`
- `_ref/*.cpp`（仅看了 RoPE 调用点）
- 15+ 个历史探针程序
- OpenMP 线程扩展性、性能测量
- MPSS/SSH 部署脚本与 COI 路线
- 长上下文行为（`pos > 5` 或 `ctx > 256` 都未测试）
- 超过 7 个构造样本的加载器模糊测试
- 未量化 PQ2_0 相对官方 logits 的残余量化误差


---

## 一、已应用的修复

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `pq2_0.h:325, 391` | **tail block 寻址差一个 block**：行步长是 `(n/128)*34`，而 tail 用 `(j/128)*34`，`j≥nn` 时 `j/128 == n/128` —— 读到**下一行的第一个 block**；最后一行则越界 | 改用向上取整步长 `rb = (n+127)/128*34` |
| 2 | `pq2_0.h:311` | 注释称内核"默认禁用"，但 `#ifndef` 实际**启用了**它（且无任何构建脚本定义该宏） | 注释改为与代码一致 |
| 3 | `gguf.c`（审计代理改） | ① 元数据值在边界检查**之前**读取；② `selftest` 的求和可回绕导致误报成功；③ 张量载荷从不校验是否超出 `file_size` | 三处均加边界/溢出检查 |

**验证**：宿主 `zig` 编译 exit 0；k1om `icc` 编译 exit 0；退路编译 exit 0；
宿主 `test_forward` **ALL CHECKS PASSED**（精确匹配 11751）；k1om `test_gguf` **41/41 通过**。

### 修复 1 的证据（审计代理用守卫页实测）
```
n=200, d=2, 缓冲区恰好 68 字节且紧贴 PAGE_NOACCESS
  ref_tail   → exit 0xC0000005（访问违例）
  tail_dep   → 只改 row1 的第一个 block，row0 的结果就变了
  ceil_layout→ row0 正确，row1 错误
```
**今天不可达**（所有 n 都是 128 倍数），但代码留着就必须正确。

---

## 二、审计纠正了我一个错误假设 ⚠️

**我一直以为 k1om 的 `long` 是 32 位**，并据此在编译命令里加了
`-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE`，还在文档里当成关键坑记录。

**实测证明我错了：k1om 是 LP64。**

```
在 mic0 上用 icc -mmic 编译运行探针，读一个 3 GB 文件：
  sizeof(long)  = 8
  sizeof(off_t) = 8
  fseeko/ftello 返回 3145728000 == stat
  去掉 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE 行为完全一致
k1om sysroot 的 bits/wordsize.h 明确写着 __WORDSIZE 64
```

所以 32 位截断的危险在卡上**不存在**，那两个宏是多余的（无害）。
这个错误假设来自我对"32-bit `long`"的早期笔记 —— 那条笔记本身是错的，或只在别的目标上成立。

## 三、审计还纠正了我一个计算错误

**我说"最后张量后有 20,384 字节余量"，不完整。**

```
最后结束的张量: blk.63.post_attention_norm.weight (F32)
  end = 7,206,168,928 = file_size    → 距 EOF = 0 字节！
PQ2_0 里最后结束的: blk.63.ffn_up.weight
  距 EOF = 20,480 字节（+96 越界读后仍有 20,384）→ 安全
```

**真实的边界在 EOF，而 `gguf.c` 分配的就是恰好 `file_size`** —— 多读 1 字节就是堆越界。
今天靠"最后一个是 F32 张量"侥幸安全。**若换导出顺序让 PQ2_0 张量排最后，越界读就会真的越界。**

**会出问题的具体位置**：`blk.63.ffn_up.weight` 最后一行的最后一个 block
（`model.c` → `mm_folded` → `pq2_0_matvec_knc` → `pq2_0_dequant_block_imci`）。
它对**前面**也会读最多 61 字节，所以紧贴分配起点的 block 同样有风险。

---

## 四、已报告但未修的发现

### 高危（畸形输入触发，正常模型不触发）

| # | 位置 | 问题 |
|---|---|---|
| F2 | `model.c:539` | `(size_t)(ck-2)` 下溢：`ssm_d_conv==1` 时算出 `(size_t)-1*conv_dim`，向缓冲区**之前** memcpy 40 KB。`calloc(0)` 返回非 NULL，检查通过 |
| F3 | `model.c:267, 479` | 先做除法再校验：`value_dim / ssm_dt_rank`、`rep = vh/kh` 在几何检查（`:269-274`）**之前**执行，而检查里没有这两个字段 → 实测 SIGFPE（`0xC0000094`） |

### 中危

| # | 位置 | 问题 |
|---|---|---|
| F4 | `model.c:314-360` | 38 个分配里只有 8 个检查了返回值，其余 30 个 OOM 时直接解引用 NULL，而不是走干净的 "allocation failed" 路径 |
| F5 | `main_k1om.c:88-91` | 解码循环里的 forward 失败被丢弃，`:97` 只要 `best==OFFICIAL_NEXT` 就返回 0 |
| F6 | `main_k1om.c:33` | `mode` 是黑名单（任何非 `"selftest"` 的字符串都会跑 7.2 GB 的 forward 路径）；文档里写的 `[threads]` 参数**从未被解析** |

### 低危 / 信息

| # | 位置 | 问题 |
|---|---|---|
| F7 | `model.c:75-100` 等 13 处 | `dequant_to_f32` 返回值被全部忽略；`mm_folded` 在张量缺失时提前返回，**留下陈旧的 `y`**（logits 保留上一个 token 的值），而 `bonsai_forward` 仍返回 0 |
| F8 | `pq2_0.h:203` | 越界读上界写的是"约 96 字节"，实际按相位最坏是 **112 字节**（本模型最紧的 block 恰好是 96） |
| F9 | `pq2_0.h:445` | `hadamard_forward` 的 `inverse` 参数**完全没作用**（两个分支字节相同，实测 `memcmp` 一致） |
| F10 | `pq2_0.h:440` | 要求 `block | n` 且 `block > 0`，只检查了 `> 0`；实测 `n=10, block=8` 会写越界 6 个元素 |
| F11 | `model.c:228` | `read_sign_vector` 在 `width` 不存在时**仍返回成功**（读到的符号切片是错的，但在界内） |
| F12 | `model.c:170` | `mm_plain` 是死代码，且是**对齐地雷**（`signs==NULL` 时把调用者缓冲区直接交给 IMCI 加载，实测 384 次访问中 128 次未对齐） |
| F13 | `model.c:324,335` | `s->delta` 和 `s->att` 分配+释放但从未使用 |
| F14 | `model.c:393` | `bonsai_model_runtime_bytes` 漏算 13 个缓冲区（本配置约 166,944 字节），却算上了未使用的 `att` |
| F15 | 加载器 | 在数据区内部被截断的文件仍能加载成功（`data_offset+sum(nbytes)=772 < 4695`）。只有 `test_gguf.c:119` 的精确铺满断言能发现 |

---

## 五、经实测确认**正确**的部分

| 项 | 证据 |
|---|---|
| `aligned_f32` / `aligned_f32_free` | 1024 次运行：0 未对齐、0 短分配、0 簿记违规，UBSan 干净 |
| 向量化解码器的索引/LUT/移位/存储数学 | 16384 次解码覆盖全部 64 种字节相位，**与标量逐位一致** |
| 8 次 FMA 的权重覆盖 + 两个内核的 tail **切分** | 32 次运行覆盖 `n=0..5120`（含 1,127,129,257,1000）零错误；随机 `(n,d)` 40 次最坏反向误差 1.2e-07 |
| 八个 `_mm512_load_ps` 偏移 | 512 位对齐全部满足；384 次访问 0 未对齐 |
| `bonsai_fp16_to_fp32` | **全部 65536 个 half 值**与 numpy 逐位一致（次正规精确，2046 个 NaN 输入全为 NaN） |
| `fwht_inplace` == 文档的奇偶矩阵 | 63/63（n=1..32）；归一化 H 正交（相对 4.1e-9）；H²=I（2.4e-7） |
| UB 检查 | UBSan（`-fsanitize=undefined -fno-sanitize-recover=all`）在 fp16、解码器、两个内核、Hadamard、`aligned_f32` 上**全部干净** |
| 加载器 64 位寻址 | `data_offset 11120992 + sum(nbytes) == file_size`；LP64 确认 |
| 类型普查与尺寸 | F32 353 / BF16 96 / PQ2_0 402；PQ2_0 = 17 字节/128 权重 = 2.125 bpw |
| 资源释放 | 所有路径都关闭文件句柄；`bonsai_gguf_free` 无重复释放；排序先于二分查找 |
| `test_forward.c` 的 ground truth | 从 `eval_callback_full.txt` 逐项重新推导：6 个常量对应的都是 **row 0**，全部正确 |

### 关于"测试期望值过期"的疑问 —— 已排除

审计代理提出 `test_forward.c` 的最终状态期望值可能过期（它自己的探针给出比值 1.0561，
而 `GDN_AXIS_AND_L2_FINDINGS.md` 记录 0.9997）。

**我实测三次都是 `l_out-63 first3_err=4.60e-02 OK`、比值 0.9997**，与文档一致，
且同一二进制 top-1 精确匹配 11751。**期望值不过期**；那个 1.0561 应是它探针本身的问题
（很可能取了 token 0 而非 5-token prompt 的最后一个 token —— 这两个是不同的向量，
单 token 的 `l_out-63` 首 3 值是 `[-2.0233 -5.2912 0.9587]`，最后一个 token 是
`[8.2843 -4.3284 5.6489]`）。

---

## 六、建议的修复优先级

```
1. F2 (model.c:539 下溢)  + F3 (先除后校验)      ← 各改 1-2 行，消除畸形输入崩溃
2. F4 (30 个未检查的分配)                        ← 统一成一个循环检查
3. F5/F6 (main_k1om 退出码 + mode 白名单)
4. F1 已修，但建议补一个断言：n%128==0（本模型成立），否则 tail 必须正确
5. 删除根目录的旧副本，避免误编译
6. 文档更新：k1om 是 LP64（我原来的笔记是错的）；越界读上界是 112 而非 96
```

## 七、未覆盖

- 前向的数值正确性 vs Python oracle（另一审查代理只做了常量核对）
- k1om **真机**上的 IMCI 语义（审查代理用模拟 shim；无卡）
- OpenMP 行为、性能测量
- ASan（本平台链接失败，改用守卫页）
- `src/bonsai/` 下另外约 30 个文件（探针程序）
- 契约级问题：`debug_cap < 0`、`bonsai_generate` 的 `out/n_out`、`ctx == 0`
