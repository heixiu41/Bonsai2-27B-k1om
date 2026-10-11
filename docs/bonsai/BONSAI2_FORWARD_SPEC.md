# Ternary Bonsai 2 27B — 前向语义规格（从 PrismML fork 源码提取）

> 来源：`PrismML-Eng/llama.cpp` 分支 `prism`，本地副本 `D:\llama-prism-src\llama.cpp-prism`
> 目的：作为写参考实现的唯一依据。所有结论都带源码行号，可复核。
> 提取日期：2026-10-09

---

## 0. 为什么需要这份文档

模型不是常规 transformer。PrismML 在权重里做了三件非常规的事：

1. **PQ2_0 三值量化**（每 128 权重一组，含 ±2 的 2-bit 码）
2. **Hadamard 旋转基**（权重离线旋转，运行时必须对激活施加匹配变换）
3. **Gated DeltaNet 混合注意力**（48 层线性注意力 + 16 层全注意力）

第 2 条尤其危险：**不做旋转不会崩，只会静默输出乱码**。fork 里专门写了图校验器
（`src/llama-context.cpp:36` `llama_verify_hadamard_graph`）来拒绝"折叠权重被消费但没做激活变换"的图。

---

## 1. Hadamard 旋转（最重要）

### 1.1 旋转矩阵的构造

`src/llama-model.cpp:2054-2066`：

```c
const float scale = 1.0f / sqrtf((float) block_size);   // block_size = 1024 → scale = 1/32
for (row = 0; row < block_size; ++row)
  for (col = 0; col < block_size; ++col) {
      uint32_t parity = row & col;
      parity ^= parity >> 16; parity ^= parity >> 8; parity ^= parity >> 4;
      parity ^= parity >> 2;  parity ^= parity >> 1;
      R[row][col] = (parity & 1) ? -scale : scale;      // Sylvester-Walsh-Hadamard，归一化
  }
```

即 `R = H / 32`，其中 `H[row][col] = (-1)^popcount(row & col)`，`H` 满足 `H·H = N·I`，
所以 `R` 是**正交且对称**的（`R⁻¹ = Rᵀ = R`）。

### 1.2 符号向量

`prism.hadamard.sign_mode = explicit`，符号**显式存储**：

| 元数据 | 值 | 含义 |
|---|---|---|
| `sign_widths` | `[5120, 6144, 17408]` | 三个符号向量的长度 |
| `sign_values` | 28672 个 ±1 | `5120 + 6144 + 17408 = 28672` ✓ |

三个宽度恰好等于模型里三种「矩阵乘的输入维度」：

- **5120** = `embedding_length`
- **6144** = `ssm_d_inner`（SSM 的 value 维度）
- **17408** = `feed_forward_length`

按 `sign_widths` 的顺序切分：`s_5120 = values[0:5120]`，`s_6144 = values[5120:6144+5120]`，
`s_17408 = values[11264:28672]`。

### 1.3 激活侧变换的确切形式

`src/llama-graph.cpp:1569-1573`（`build_lora_mm`）：

```c
if (t.signs) { cur_mm = ggml_mul(ctx0, cur_mm, t.signs); }   // 逐元素乘符号
cur_mm = llama_mul_mat_hadamard(ctx0, cur_mm, t.rot);        // 乘旋转矩阵
```

`src/llama-impl.h:57-75`：

```c
res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);   // n = rot->ne[0] = 1024
res = ggml_mul_mat(ctx, rot, res);                            // [1024,1024] @ [1024, m]
res = ggml_reshape_4d(ctx, res, ...);
```

**结论 —— 激活变换是：**

```
x' = H · (s ⊙ x)
```

其中 `s` 是长度等于该权重**输入维度**的 ±1 向量，`H` 是 1024 分块的归一化 WHT，`⊙` 为逐元素乘。

> 推导：权重存的是 `R·W`（`R = diag(s)·H`），前向要 `(R·W)ᵀ·x = Wᵀ·Rᵀ·x = Wᵀ·(H·(s⊙x))`。
> `Rᵀ = Hᵀ·diag(s)ᵀ = H·diag(s)`，作用在 `x` 上即先按 `s` 翻符号、再做 `H`。

### 1.4 哪些张量被折叠（401 个，实测）

```
output.weight                  x1    输入 5120
blk.*.attn_qkv.weight          x48   输入 5120   ← SSM 层的融合 QKV
blk.*.attn_gate.weight         x48   输入 5120   ← SSM 层的门控
blk.*.ssm_out.weight           x48   输入 6144
blk.*.ffn_gate.weight          x64   输入 5120
blk.*.ffn_up.weight            x64   输入 5120
blk.*.ffn_down.weight          x64   输入 17408
blk.*.attn_q.weight            x16   输入 5120   ← 全注意力层
blk.*.attn_k.weight            x16   输入 5120
blk.*.attn_v.weight            x16   输入 5120
blk.*.attn_output.weight       x16   输入 6144
```

**规律：每个折叠权重的 `ne[0]`（输入维度）必然是 5120 / 6144 / 17408 之一**
，符号向量按此选取。`llama-model.cpp:2006` 会校验 `weight->ne[0] % block_size == 0`。

注意 `ffn_down` 的输入是 17408（不是 5120），所以 3 个符号向量缺一不可。

### 1.5 特例：embedding 的**逆向**变换

`prism.hadamard.inverse_weight_names = ['token_embd.weight']`

**已用官方 `llama-eval-callback` 的 dump 端到端验证**（prompt = "The capital of France is"，
token `[760,6511,314,9338,369]`）。官方图里嵌入路径的实际算子序列：

```
node_0            = GET_ROWS(token_embd.weight, inp_tokens)      {5120, 5, 1, 1}
(reshaped)        = RESHAPE(node_0)                              {1024, 25, 1, 1}
node_2            = MUL_MAT(prism.hadamard.1024, (reshaped))     {1024, 25, 1, 1}
(reshaped)        = RESHAPE(node_2)                              {5120, 5, 1, 1}
model.input_embed = MUL((reshaped), prism.hadamard.signs.5120)   {5120, 5, 1, 1}
norm-0            = RMS_NORM(model.input_embed)
attn_norm-0       = MUL(norm-0, blk.0.attn_norm.weight)
```

**两条容易搞错的结论：**

1. **逆向变换也要乘符号，且顺序是「先 H、后乘符号」** ——
   `x' = s ⊙ (H x)`，而不是我原先以为的裸 `H x`。
2. **H 按 1024 分块，符号按整行（5120）** —— 两者的分组粒度不同。
   `apply_hadamard` 里的矩阵元素已含 `1/sqrt(block)`，**不要再额外归一化**
   （我一度多除了一次 32，导致整体差 32 倍）。

> ⚠️ **解析 dump 的陷阱**：`common_debug_print_tensor` 按 `n=3` 打印，
> 每个维度**首 3 + 末 3**共 6 个值（`%12.4f`）。当 `ne[1]=5`（5 个 token）时，
> 每个 token 各占一行，`[... ]` 与 `[..., ..., ... ]` 之间是省略号。
> **把末 3 值当成第 4-6 个值是最容易犯的解析错误** —— 我犯过，
> 由此得出"前 3 个对、后面发散"的错误结论，实际上 5 个 token 全部精确吻合。

**验证结果**（我的 numpy 实现 vs 官方 dump，5 个 token 的首/末 3 值）：

| token | 官方 first3 | 我的 first3 | 官方 last3 | 我的 last3 |
|---|---|---|---|---|
| 760 | `0.0178 0.0010 0.0039` | `0.0178 0.0010 0.0039` ✓ | `0.0009 0.0034 0.0081` | `0.0009 0.0034 0.0081` ✓ |
| 6511 | `0.0172 0.0100 -0.0269` | 一致 ✓ | `-0.0022 -0.0047 0.0138` | 一致 ✓ |
| 314 | `0.0113 0.0038 0.0016` | 一致 ✓ | `-0.0002 0.0083 0.0028` | 一致 ✓ |
| 9338 | `-0.0016 0.0163 0.0373` | 一致 ✓ | `0.0024 -0.0230 0.0210` | 一致 ✓ |
| 369 | `-0.0035 0.0129 -0.0017` | 一致 ✓ | `-0.0009 -0.0006 -0.0025` | 一致 ✓ |

`attn_norm-0` 首值 `1.3691` 也已核对：`sqrt(5120)=71.554`，
`0.017833 × 71.554 × 1.0735 ≈ 1.369` ✓

### 1.7 阶段比对结果（`scripts/bonsai_diff_gt.py`）

用官方 `llama-eval-callback` 的 dump 做逐阶段比对，阈值取 `2e-4`（dump 显示到 4 位小数）：

| 阶段 | 最差误差 | 判定 | 说明 |
|---|---|---|---|
| `attn_norm-0` | 5.0e-05 | **通过** | 纯 F32 权重 → 位级吻合 |
| `node_17`（×signs） | — | **通过** | 首尾值完全一致 |
| `qk_conv_l2-0` 范数 | — | — | — |
| `linear_attn_qkv_mixed-0` | 7.6e-03 | 偏差 | **PQ2_0 权重** |
| `z-0` | 7.2e-03 | 偏差 | **PQ2_0 权重** |
| `conv_output_silu-0` | 3.7e-04 | 偏差 | 受上游传导 |
| `new_state-0` | 1.9e-03 | 偏差 | 受上游传导 |
| `final_output-0` | 4.3e-01 | 偏差 | 受上游传导 |
| `linear_attn_out-0` | 1.5e+00 | 偏差 | 受上游传导 |

**结论：分歧点精确定位在「PQ2_0 量化权重的矩阵乘」这一步。**

- Hadamard 链本身已证明正确：`attn_norm` 通过、`node_17` 通过、
  正交性验证 `L2 75.964 → 75.964`（保持范数）
- 纯 F32 权重的算子（`attn_norm`、`RMS_NORM`）误差 5e-5，即 dump 的显示精度
- **凡是用 PQ2_0 权重的算子都带 ~0.25% 的相对偏差**
  （`7.6e-3 / 3.0 ≈ 0.25%`），与 2.13 bpw 三值量化的精度量级一致

下游的大误差（`linear_attn_out` 1.5）是**上游传导 + 放大**，不是独立缺陷。

> **待确认**：这 0.25% 究竟是**量化本身的固有损失**，还是我的 PQ2_0 解码有一处小错
> （例如 `code 3 → +2` 的映射、或 fp16 scale 的一个 LSB）。
> 区分方法：拿 `prism-ml/Ternary-Bonsai-27B-unpacked`（官方解包版）对比我的
> `dequant()` 逐元素输出。在确认之前，**不应宣称前向完全正确**。

### 1.8 加载器的独立交叉验证

用官方 `llama-gguf.exe <file> r` 读取头部，与我的解析器逐项对照：

| 项 | 官方工具 | 我的解析 |
|---|---|---|
| version | 3 | 3 |
| alignment | 32 | 32（`align32(info_end)`） |
| **data offset** | **11120992** | **11120992** ✓ |
| n_kv | 49 | 49 |
| n_tensors | 851 | 851 |

数据区起点**逐字节吻合**，独立确认了 §1.1 的偏移基准推断。

### 1.6 不生效的部分

`prism.hadamard.gdn_v_grouped` **不在**元数据中（49 个键里没有）→
`llama-model.cpp:2123` 的 `ssm_out` 额外 head 重排 **不生效**。实现时可忽略 `perm_hd/perm_nk/perm_rep`。

---

## 2. Gated DeltaNet（48 个线性注意力层）

### 2.1 该层的张量与维度

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `attn_norm.weight` | `[5120]` | F32 | RMSNorm |
| `attn_qkv.weight` | `[5120, 10240]` | PQ2_0 | 融合 QKV |
| `attn_gate.weight` | `[5120, 6144]` | PQ2_0 | 输出门控 `z` |
| `ssm_alpha.weight` | `[5120, 48]` | BF16 | 对数衰减率 |
| `ssm_beta.weight` | `[5120, 48]` | BF16 | 学习率 |
| `ssm_a` | `[48]` | F32 | `-A_log.exp()`，非递归参数 |
| `ssm_dt.bias` | `[48]` | F32 | alpha 的偏置 |
| `ssm_conv1d.weight` | `[4, 10240]` | F32 | **注意形状是 [k, C]，通道 10240 = QKV 维度** |
| `ssm_norm.weight` | `[128]` | F32 | 门控归一化（作用在 head 维） |
| `ssm_out.weight` | `[6144, 5120]` | PQ2_0 | 输出投影 |

**维度关系**（`src/models/qwen35.cpp:59-65` + `:81-89`，权威定义）：

```c
head_k_dim = head_v_dim = ssm_d_state = 128
n_k_heads  = ssm_n_group    = 16      → key_dim   = 128*16 = 2048
n_v_heads  = ssm_dt_rank    = 48      → value_dim = 128*48 = 6144
conv_dim   = key_dim*2 + value_dim    = 4096 + 6144 = 10240
```

张量形状（实测与建图代码逐一对上）：

| 建图代码 | 实测 dims | 校验 |
|---|---|---|
| `wqkv {n_embd, key_dim*2+value_dim}` | `[5120, 10240]` | ✓ |
| `wqkv_gate {n_embd, value_dim}` | `[5120, 6144]` | ✓ |
| `ssm_conv1d {conv_kernel, conv_dim}` | `[4, 10240]` | ✓ |
| `ssm_beta/alpha {n_embd, n_v_heads}` | `[5120, 48]` | ✓ |
| `ssm_a / ssm_dt.bias {n_v_heads}` | `[48]` | ✓ |
| `ssm_norm {head_v_dim}` | `[128]` | ✓ |
| `ssm_out {value_dim, n_embd}` | `[6144, 5120]` | ✓ |

> ⚠️ **极易搞混的一点**：这一层里 **q 和 k 各只有 2048 维（16 头）**，
> 而 **v 有 6144 维（48 头）**。`num_v_heads/num_k_heads = 48/16 = 3`，
> 即每个 k 头被 3 个 v 头共享（GQA 式）。看到 `10240 = 2048*2 + 6144` 就不会再误判。

> 混淆来源：`delta-net-base.cpp:474` 里的 `conv_channels = d_inner + 2*ssm_n_group*ssm_d_state`
> 是**另一个函数**（generic mamba 路径）的式子，数值上等于 6144+4096=10240 恰好巧合相同；
> qwen35 走的是 `delta-net-base.cpp:454 build_conv_state`，直接用 `conv_dim = 10240`。

`build_conv_state` 把 3 个历史时间步的 QKV 与当前 QKV 拼接后做因果卷积
（`delta-net-base.cpp:471-478`），**卷积只作用于 QKV 的 10240 维，门控 `z` 不参与卷积**。

### 2.2 逐 token 前向

参考 `src/models/qwen35.cpp:408-582` 与 `src/models/delta-net-base.cpp:289-371`。

```c
// ---- 1. 输入投影（cur 是 attn_norm 之后的激活，长 5120）----
qkv = W_qkv @ hadamard(cur)        // [1, 10240]
z   = W_gate @ hadamard(cur)       // [1, 6144]

beta_raw  = W_beta  @ cur          // [1, 48]   BF16
alpha_raw = W_alpha @ cur          // [1, 48]   BF16
beta  = sigmoid(beta_raw)                              // qwen35.cpp:437
alpha = softplus(alpha_raw + ssm_dt_bias)              // qwen35.cpp:460-461
g     = alpha * ssm_a                                  // = -A_log.exp()*softplus   :464

// ---- 2. 因果卷积（kernel=4，分块深度卷积）----
// conv 输入 = [4 个历史 token 的 qkv] 拼当前 qkv，共 4 个时间步
conv_out = ssm_conv1d ⊛ qkv            // 含状态的因果卷积，通道数 10240
conv_out = silu(conv_out)                              // :502

// ---- 3. 切出 q/k/v ----
q = conv_out[0*16*128 : 1*16*128]      // [16, 128]  k 头
k = conv_out[1*16*128 : 2*16*128]      // [16, 128]
v = conv_out[2*16*128 : 3*16*128]      // [48, 128]  v 头，长 6144

// ---- 4. q,k 的 L2 归一化（不是 RMSNorm！）----
// qk 是相邻的两段同宽 head 组，先拼成 [32,128] 一起做 L2 归一化，再切开
qk = concat(q, k, axis=head);  qk = l2_norm(qk, eps_norm);       // :543
q, k = split(qk)

// ---- 5. k 头 -> v 头的广播 ----
// num_v_heads/num_k_heads = 3，q/k 各重复 3 次以对齐 v
if (num_v_heads != num_k_heads) { q = repeat(q, 3); k = repeat(k, 3); }   // :559-560

// ---- 6. Gated DeltaNet 递推（每 token 一步；state S 形状 [48][128][128]）----
scale = 1/sqrt(128)
q *= scale
Sk = exp(g)                            // 标量衰减（每 v 头一个，48 个）
S  = S * Sk                            // [128,128] 逐元素衰减
sk = rowsum(S * k_h)                   // Sᵀk -> [128]
d  = (v_h - sk) * beta_h               // [128] 误差项
S  = S + outer(k_h, d)                 // 状态更新
o_h = rowsum(S * q_h)                  // Sᵀq -> [128] 输出

// ---- 7. 门控归一化 + 输出投影 ----
out = gated_norm(o, z, ssm_norm)       // :574  build_norm_gated
out = W_ssm_out @ hadamard(out)        // [1, 5120]
```

**DeltaNet 递推的精确形式**（`delta-net-base.cpp:319-366`，`S` 是 128×128 矩阵）：

```
S ← S ⊙ exp(g)                          // 行 340（g 广播到整个矩阵）
sk = Sᵀ k                               // 行 344-345
d  = β (v − sk)                         // 行 349-350
S ← S + k ⊗ d                           // 行 359-361
o  = Sᵀ q                               // 行 365-366
```

注意 `s = s * g` 前先 `g = exp(g)`（行 339-340），且 `g` 在该层被 reshape 成 `[1, g_dim, H_v, B]`，
即**每个 v 头一个标量衰减**（GDA 情形），不是逐元素。

### 2.3 需要留意的点

- `q,k` 用的是 **L2 归一化**（`ggml_l2_norm`），不是 RMSNorm，且**不乘权重**
- `beta` 走 **sigmoid**；`alpha` 走 **softplus** 后加 `ssm_dt.bias`，再乘 `ssm_a`
- 卷积是**含历史状态的因果卷积**（`build_conv_state`，`delta-net-base.cpp:454`），
  状态缓存长度 `conv_kernel - 1 = 3`
- `ssm_conv1d.weight` 是 F32 且形状 `[4, 10240]`，注意 ggml 的 `ne[0]=4` 是**卷积核长度**

---

## 3. 全注意力层（16 个，`i % 4 == 3`）

参考 `src/models/qwen35.cpp:327-406`。

| 张量 | 形状 | 说明 |
|---|---|---|
| `attn_q.weight` | `[5120, 12288]` | **Q 与门控融合**（`2*256*24`） |
| `attn_k.weight` | `[5120, 1024]` | 4 个 KV 头 × 256 |
| `attn_v.weight` | `[5120, 1024]` | |
| `attn_q_norm.weight` | `[256]` | QK-norm（head 维 256） |
| `attn_k_norm.weight` | `[256]` | |
| `attn_output.weight` | `[6144, 5120]` | 输入 6144 = 24 头 × 256 |

```c
Q_full = W_q @ hadamard(cur)          // [12288]
Q    = Q_full[0 : 24*256]             // 前一半
gate = Q_full[24*256 : 2*24*256]      // 后一半，是门控
Q = rmsnorm_head(Q, attn_q_norm)      // 每头 256 维做 RMSNorm     :348
K = rmsnorm_head(K, attn_k_norm)      // :359
Q,K = rope_multi(Q,K, sections=[11,11,10,0], freq_base=1e7, n_rot=64)   :372-381
attn = softmax(QKᵀ/√256) @ V          // GQA：24 q 头 / 4 kv 头
attn = attn * sigmoid(gate)           // 门控
out = W_out @ hadamard(attn)          // [6144] -> [5120]
```

**要点**：
- `head_dim = 256`，但 **`n_rot = 64`**（`rope.dimension_count`）→ **RoPE 只作用于每头前 64 维**
- `rope.dimension_sections = [11, 11, 10, 0]` 是 **M-RoPE**（多模态分段），
  `sum = 32`，`n_rot = 2 * 32 = 64` ✓
- Q 投影输出 12288 = **Q(6144) 与门控(6144) 拼接**，别当成 48 头
- V 是 4 个 KV 头 → 1024 = 4×256，按 GQA 广播给 24 个 Q 头

#### RoPE 的配对方式（已对源码验证）

`ggml-cpu/ops.cpp:6090-6093` 对 `MROPE` 走 `rotate_pairs(n_dims, n_offset = n_dims/2, scale = 2)`：

```c
// ops.cpp:5944-5959
for (i0 = 0; i0 < n; i0 += 2) {
    ic = i0 / scale;                    // scale = 2 → ic = i0/2
    cos_theta = cache[i0]; sin_theta = cache[i0+1];
    x0 = src[ic];  x1 = src[ic + n_offset];      // 配对 (i, i + n_dims/2)
    dst[ic]           = x0*cos - x1*sin;
    dst[ic + n_offset] = x0*sin + x1*cos;        // 同一个 θ_i 作用于两个成员
}
// ops.cpp:6024
theta_scale = powf(freq_base, -2.0f / n_dims);
```

即 **NeoX 式前后半配对**（不是相邻配对），θ_i = `pos * freq_base^(-2i/n_dims)`，i ∈ [0, 32)。

> ⚠️ **唯一未验证项**：元数据里**没有** `mrope_section` / `interleaved` 开关，
> 只有 `dimension_sections`。对**纯文本**输入，ggml 的 `theta_t/theta_h/theta_w` 取自
> `inp_pos` 的不同行；当所有行是同一个位置索引时三者相同，于是退化为标准 NeoX RoPE
> —— 这也是参考实现 `rope_multi` 所做的（已验证配对与 theta_scale 一致）。
> **多模态（图像）输入需要三个不同的位置索引，此路径未实现也未验证。**

---

## 4. 公共部分

- 所有层：`attn_norm`(RMSNorm, eps=1e-6) → 注意力 → 残差 → `post_attention_norm`(RMSNorm) → FFN → 残差
  （`qwen35.cpp:198-229`）
- FFN 是 SwiGLU：`down(silu(gate(x)) * up(x))`，`silu = x*sigmoid(x)`
- 最终 `output_norm`(RMSNorm) → 输出头
- `pipeline`：`embedding_length=5120`，`n_head=24`，`n_head_kv=4`，`head_dim=256`

---

## 5. 验证策略

参考实现的正确性只能用**逐 token 对齐**来证明。可行做法：

1. 用 PrismML fork 在**本机 CPU**（或任意能跑的设备）跑 `llama-cli`，抓：
   - 指定 prompt 的完整输出 token 序列
   - 首步 logits（fork 支持 `--logits-file` 之类的导出，或用 `llama-perplexity`）
2. 自己的实现跑同一 prompt，比对：
   - **首步 top-1 token 必须一致**
   - logits 相对误差在 Q8/Q2 噪声量级内
   - 逐 token 生成序列一致（贪婪采样、固定 seed）
3. 分层排查：fork 有 `cb()` 回调能把中间张量 dump 出来
   （如 `conv_output_silu`、`qk_conv_l2`、`dnet_add_ar_state`、`decay_mask`），
   逐个和自己算的对比，能精确定位到哪一步发散

**没有第 1 步的参考输出，就不要上卡调。** 64 层里任何一处符号/顺序错误都会静默变成乱码，
而乱码不提供任何定位信息。

---

## 6. 源码索引

| 文件 | 用途 |
|---|---|
| `src/models/qwen35.cpp:408-582` | GDN 层完整前向 |
| `src/models/delta-net-base.cpp:289-371` | **DeltaNet 单 token 递推（最关键的参考）** |
| `src/models/delta-net-base.cpp:16-287` | 分块（chunked）实现，理解数学等价性 |
| `src/llama-graph.cpp:1546-1603` | `build_lora_mm`，Hadamard 激活变换的施加点 |
| `src/llama-impl.h:57-75` | `llama_mul_mat_hadamard` |
| `src/llama-model.cpp:1972-2139` | 旋转矩阵构造 + 符号表装配 |
| `src/llama-model.cpp:1230-1265` | 符号元数据解析与校验 |
| `src/llama-context.cpp:36-110` | **Hadamard 图校验器**（看清哪些消费方式是非法的） |
| `ggml/src/ggml-common.h` | `block_pq2_0` / `block_ptq1_0` 定义 |
| `ggml/src/ggml-quants.c:474-540` | `dequantize_row_q2_0` / `dequantize_row_pq2_0` |
