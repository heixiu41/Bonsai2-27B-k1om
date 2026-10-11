# Ternary Bonsai 2 27B → Xeon Phi 7120P (k1om) 移植计划与验证状态

> ## ⚠️ 本文件是**早期计划文档**，多处已过时
>
> **先读 `HANDOVER.md`。** 本文件保留作历史记录，但以下几处**已知是错的**：
>
> | 位置 | 错在哪 |
> |---|---|
> | §6.3 环境陷阱 | 把"`ftell()` 是 32 位 `long`"列为关键坑 —— **k1om 是 LP64**，`sizeof(long)==8`，实测确认。该危险只存在于 i386 之类的 ILP32 目标 |
> | §5.4 / §6 | "预期 1–4 tok/s"、"实测有效带宽 49 GB/s" —— 重做基准后 `rows-1360` 模式达 **196.43 GB/s**，访存只占每 token 时间的 **1%** |
> | §6.2 / 越界读 | 写"约 96 字节"，守卫页实测是 **48 字节** |
> | IMCI 内核状态 | 曾写"默认禁用"；现在是 k1om **默认启用**，且那个"结果恒为一半"的缺陷已修 |
> | 根目录副本 | 文件里若提到根目录的 `pq2_0.h`/`model.c`，那套陈旧副本**已删除** |
>
> 当前权威状态、构建配方、验收标准、以及我犯过的错，都在 **`HANDOVER.md`**。

> 目标：让 PrismML 的 Ternary Bonsai 2 27B 在 Intel Xeon Phi 7120P（k1om）上原生运行。
> 硬件：61 核 / 244 线程，16 GB GDDR5，352 GB/s 理论带宽；`mic0` 在线，IP 192.168.1.100。
> 状态日期：2026-10-09
>
> **文档说明**：本文件曾因使用 PowerShell `Set-Content` 写入而遭编码损坏（不可逆），
> 于第 11 轮由已验证内容重建。**教训见文末「编码陷阱」一节** —— 含非 ASCII 的文件
> 只能用 `write`/`edit` 工具修改，绝不用 PowerShell 写回。

---

## 1. 模型与架构（实测，非推断）

Ternary Bonsai 2 27B = **Qwen3.8-27B 的端到端三值压缩版**，5.95 GB（PTQ1_0）/ 7.21 GB（PQ2_0）。

**它不是常规 transformer。** GGUF 元数据实测（49 个键全部解析）：

```
general.architecture         = qwen35
block_count                  = 64          embedding_length = 5120
feed_forward_length          = 17408       （SwiGLU）
attention.head_count         = 24          head_count_kv    = 4
attention.key_length         = 256         value_length     = 256
rope.dimension_count         = 64
rope.dimension_sections      = [11, 11, 10, 0]     ← M-RoPE
rope.freq_base               = 1e7
attention.layer_norm_rms_epsilon = 1e-6
ssm.conv_kernel              = 4           ← Mamba 式状态空间
ssm.state_size               = 128
ssm.group_count              = 16
ssm.time_step_rank           = 48
ssm.inner_size               = 6144
full_attention_interval      = 4           ← 每 4 层 1 层真注意力
prism.hadamard.block_size    = 1024
prism.hadamard.weight_names  = 401 个张量
prism.hadamard.sign_widths   = [5120, 6144, 17408]
prism.hadamard.inverse_weight_names = ['token_embd.weight']
tokenizer                    = gpt2 / pre=qwen35，词表 248320
```

**层结构**：64 层 = **48 个 GDN（Gated DeltaNet，线性注意力）层 + 16 个全注意力层**（`i%4==3`）。
851 个张量（402 个 PQ2_0 + 353 F32 + 96 BF16）。

---

## 2. 二进制格式（已逐项实测 / 交叉验证）

### 2.1 GGUF 头

| 项 | 值 | 验证方式 |
|---|---|---|
| version | 3 | 官方 `llama-gguf.exe r` |
| alignment | 32 | 同上 |
| **data offset** | **11120992** | 官方工具与本实现**逐字节一致** |
| n_kv / n_tensors | 49 / 851 | 同上 |

**张量 `offset` 是相对于数据区起点的偏移**，基准 = `align32(元数据块 + 张量信息块结束位置)`。
当成绝对文件偏移会落进错误张量且**不报错**，只静默解出垃圾（F32 RMSNorm 权重解出 `1e+25`）。

**维度约定**：`dims[0]` 是**最内层（输入）维度**，与 numpy 相反。
例如 `attn_qkv` 的 `dims={5120, 10240}` 表示输入 5120、输出 10240。

自洽性硬验证：`data_offset + Σ nbytes == file_size`（**误差 0 字节**）。

### 2.2 PQ2_0 量化（选它而非 PTQ1_0）

```c
#define QK_PQ2_0 128
typedef struct { ggml_half d; uint8_t qs[32]; } block_pq2_0;   // 34 B = 2.125 bpw
// 码 → 值： 00=-1, 01=0, 10=+1, 11=+2   （即 (q-1)*d）
```

选择理由：PTQ1_0 把 5 个三值塞进 1 字节、需三进制余数递推 `v=(v*3)&0xFF` 且元素顺序非位置顺序；
PQ2_0 是固定 2-bit 槽位。KNC 无整数 SIMD 与三元点积指令，标量解码下差距被放大。

**实测**：`blk.0.ffn_gate.weight` 是纯三值（码 {0,1,2} 各约 33%，无 code 3）；
attention 张量有约 0.8% 的 `+2` 与偶发大 scale（最高 ~3400）—— 是数据中真实的离群组，
每块独立 scale + 统一的 `(q-1)*d` 即可正确解码。

### 2.3 Hadamard 旋转（最关键、最易错）

权重存放在**旋转基**下。不做激活变换**不会崩，只会静默输出乱码**。
官方图的实际算子序列（`llama-eval-callback` dump 实测）：

```
node_0            = GET_ROWS(token_embd.weight, inp_tokens)
(reshaped)        = RESHAPE(node_0)            {1024, n/1024}
node_2            = MUL_MAT(prism.hadamard.1024, (reshaped))
model.input_embed = MUL((reshaped), prism.hadamard.signs.5120)     ← 逆变换也要乘符号
```

**三条结论**：

1. **正向**（401 个折叠权重的输入侧）：`x' = H · (s ⊙ x)`，
   `H[row][col] = (parity(row&col) ? -1 : +1) / sqrt(1024)`（归一化 Sylvester-WHT）。
2. **逆向**（embedding lookup 之后）：`x' = s ⊙ (H x)` —— **顺序相反**。
3. **H 按 1024 分块，符号按整行（5120/6144/17408）** —— 两种分组粒度不同。
   `H` 的元素已含 `1/sqrt(block)`，**不要再额外归一化**。

401 个折叠权重按输入维分三类：`{5120: 273, 6144: 64, 17408: 64}`。

### 2.4 `ssm_conv1d` 的内存布局（易错）

张量声明 `dims={4, 10240}`，但数据实际按 **`flat[c*conv_k + k]`**
（通道慢变、kernel 快变）存储。按 `[k*conv_dim + c]` 索引会读到**转置后的权重**。

实测证据：C 读到的 `flat[3] = -0.080566`，正是 numpy 的 `kern[0, 3]`（通道 0 的槽位 3）。
修正后 `conv` 阶段误差从 `1.29e-01` 降到 `2.74e-04`。

---

## 3. 前向语义（从 fork 源码提取，带行号）

### 3.1 GDN 层（48 层）

维度（`src/models/qwen35.cpp:59-65` 权威定义）：

```
head_k_dim = head_v_dim = ssm_d_state = 128
n_k_heads  = ssm_n_group    = 16   → key_dim   = 2048
n_v_heads  = ssm_dt_rank    = 48   → value_dim = 6144
conv_dim   = key_dim*2 + value_dim = 10240
```

前向（`qwen35.cpp:408-582`）：

```
xb   = RMSNorm(x, attn_norm)
qkv  = W_qkv @ had(xb)        [10240]
z    = W_gate @ had(xb)       [6144]
beta = sigmoid(W_beta  @ xb)              [48]
alpha= softplus(W_alpha @ xb + ssm_dt)    [48]
g    = alpha * ssm_a                       （ssm_a = -A_log.exp()，实测全为负）
conv = silu( depthwise_causal_conv(qkv, kernel=4) )
q,k  = split(conv[0:2*key_dim]) 后做【一次共享的 L2 归一化】
v    = conv[2*key_dim : ]
k,q 重复 3 次以对齐 48 个 value head
S ← S·exp(g);  sk = Sᵀk;  d = β(v − sk);  S += k ⊗ d;  o = Sᵀq
out  = W_ssm_out @ had( silu(z) ⊙ RMSNorm_perhead(o, ssm_norm) )
```

### 3.2 官方 CPU 内核的精确递推（`ggml-cpu/ops.cpp:11015-11089`）

```c
const float scale = 1.0f / sqrtf((float) S_v);

// raw_gates 为真时 beta/g 在融合算子内部激活：
//   beta_val = 1/(1+expf(-beta_val));
//   x  = g0 + raw_dt_bias[h];
//   g0 = raw_a[h] * ((x > 20) ? x : logf(1+expf(x)));

// 状态按【转置】存储：s_out[j*S_v + i] == S[i][j]
ggml_vec_scale_f32(S_v*S_v, s_out, expf(g0));      // 标量衰减
for (j) delta[j] = (v_d[j] - dot(row_j(M), k_d)) * beta_val;
for (i,j) M[j*S_v + i] += delta[j] * k_d[i];
```

### 3.3 全注意力层（16 层）

```
Q_full = W_q @ had(xb)                 [12288] = Q(6144) | gate(6144)
Q = RMSNorm_perhead(Q, attn_q_norm);  K = RMSNorm_perhead(K, attn_k_norm)
Q,K = M-RoPE(n_rot=64, sections=[11,11,10,0], base=1e7)
attn = softmax(QKᵀ/√256) @ V           （24 q 头 / 4 kv 头，GQA）
out  = W_out @ had( attn ⊙ sigmoid(gate) )
```

RoPE 配对已验证（`ops.cpp:6090` 走 `rotate_pairs(n_dims, n_offset=n_dims/2, scale=2)`）：
**NeoX 前后半配对** `(i, i+32)`，同一 θ_i 作用于两个成员，`theta_scale = freq_base^(-2/n_dims)`。

### 3.4 FFN（64 层共享）

```
xb = RMSNorm(x, post_attention_norm)
out = W_down @ had( silu(W_gate @ had(xb)) ⊙ (W_up @ had(xb)) )
```

---

## 4. 验证状态

### 4.1 已通过（对齐到官方 dump 的显示精度 4 位小数）

| 阶段 | 误差 | 备注 |
|---|---|---|
| GGUF 加载器 | — | **41 项自测全部通过**，与官方工具逐字节一致 |
| `embed`（GET_ROWS + H + signs） | 3.4e-05 | 5 个 token 首尾值全对 |
| `attn_norm` | 4.5e-05 | |
| `signed`（= node_17） | 4.5e-05 | |
| `conv_output_silu` | **2.74e-04** | 布局修复后 |
| `qk_conv_l2`（共享 L2 归一化） | 2.4e-04 | |
| `q_conv_predelta` / `k_conv_predelta` / `v_conv_predelta` | 吻合 | head 0 row 0 |
| FWHT vs 稠密 Hadamard | 3e-07 | Python 侧等价性证明 |
| PQ2_0 编解码 vs 标量参考 | 0 | |
| fp16 解码 vs numpy | 0 | 含次正规 |

**即：卷积 + SiLU + L2 归一化 + q/k/v 切分（含 3 倍广播）整条链路已完全对齐。**

**第 12 轮补充确认**（单 token dump）：

| 张量 | 我的 | 官方 | 判定 |
|---|---|---|---|
| `z-0`（`W_gate @ had(attn_norm)`） | `[-0.033036  -3.213144  -2.72947]` | `[-0.0342  -3.2155  -2.7293]` | **吻合** |
| `final_output-0` | `[-0.000104  -0.000753  -0.001936]` | `[0.0  -0.0027  -0.0028]` | 4 位小数下无法分辨 |

`final_output` 与 `z` 均已是**闸门归一化之后**的量（`z-0` 是 gate 投影、
`final_output-0` 是 `attn_out_norm` 被 reshape 成 `[128,48,1,1]`），
因此 `z` 的吻合说明闸门投影一路正确。

> **判定工具已到极限**：`final_output` 的官方打印值 `0.0` 与我的 `-0.000104`
> 在 `%12.4f` 下不可区分，`new_state` 又因是累积状态不可用。
> 因此改用**端到端信号**（见 4.4）作为此阶段的判据。

### 4.2 待解决

| 项 | 说明 |
|---|---|
| **残差量级放大（根因，见 4.5）** | 我的 `l_out-0` 已达 40，官方仅 ~0.05；到第 63 层我 `\|\|x\|\| = 1009`，官方 ~5 |
| `qkv` 的 0.2-0.6% 偏差 | PQ2_0 量化固有量级，**已排除为根因**（见 4.4） |
| 全注意力层 / 输出头 | 未对齐 |
| 端到端 top-1 | C 与 numpy 都得 248046，官方是 **11751（' Paris'）** |

### 4.5 根因定位：量级在第 0 层下游就已放大

**第 12 轮的决定性发现。** 此前一直用「逐阶段首尾值对齐」验证，所有阶段都通过，
**却从未检查量级**。加入最终层与范数对照后暴露问题。

**先纠正一个我自己造的错误**：第一版对照表用解析器取「末 3 值」，
但多 token 张量的「末 3 值」属于**最后一个 token**，不是 token 0 —— 表格因此失真。
改用**手工核对的原始值**重算后：

| 阶段 | 官方 6 值 L2 | 官方全维折算 | 我的 | 比值 | 判定 |
|---|---|---|---|---|---|
| `attn_norm-0` | — | 44.20 | 75.96 | 1.72 | 尚可 |
| `linear_attn_qkv_mixed-0` | **3.501** | 102.27 | 357.53 | — | 6 值 L2 我=**3.5013**，**吻合** |
| `final_output-0` | 0.0039 | **0.114** | **29.31** | **~258** | **量级严重偏离** |
| `linear_attn_out-0` | 0.0553 | 1.616 | 28.47 | 17.6 | 偏离 |
| `ffn_swiglu-0` | 0.0164 | 0.480 | 26.88 | 56 | 偏离 |
| `l_out-0` | 0.0866 | 2.531 | 72.88 | 28.8 | 偏离 |

**关键点**：

1. `qkv` 的范数与采样值**都吻合**（6 值 L2 分别 3.5013 / 3.5009）——
   再次确认 GDN 的输入侧完全正确。
2. 但 `final_output` 的官方采样值里 **5 个是 `0.0000`**，而我对应位置是 **29.3**。
   `0.0000` 在 `%12.4f` 下代表真值 < 5e-5 —— **与 29.3 不可能是同一量级**，
   这**不是采样误差**。
3. 官方 `final_output-0` 的整体量级（全维折算 0.114）说明这一层输出**非常小**；
   而我的输出大了约 250 倍。

**因此根因锁定在 `final_output` 之前、`qkv`/`conv`/`qk_l2` 之后**，即
**DeltaNet 状态更新（`S`）或逐头门控归一化**这一步。

官方各层残差 `l_out-N` 的演化作为旁证（首 3 值）：

| 层 | first3 | 量级 |
|---|---|---|
| 0 | `[-0.0004,  0.0492, -0.0335]` | ~0.05 |
| 15 | `[-0.2919,  0.2705,  0.1752]` | ~0.3 |
| 47 | `[-0.5254,  0.6057, -0.7174]` | ~0.7 |
| 63 | `[-2.0233, -5.2912,  0.9587]` | ~5 |

官方从 0.05 缓慢长到 5（约 100 倍，正常）。我的 `l_out-63` 却是
`first3 = [2.91  5.51  25.38]`、`||x|| = 1009.6` —— **放大约 200 倍**。

> **方法论教训**：只对齐「首尾 3 个采样值」会漏掉**整体尺度错误** ——
> 尺度错误在少数采样点上可能表现为"数值相近"，容易被当成数值噪声。
> **必须同时检查每个阶段的范数/量级**，而不只是逐点比对。
> 这次的教训更深一层：**连范数对比也要用原始值手算**，
> 因为解析器取「末 3 值」时同样会取错行（见 6.2）。

### 4.4 端到端判据：官方 token 在 logits 中的排名

中间量精度不够时，用**官方期望 token（11751）在我方 logits 中的排名**来判断
前向"接近但有小错"还是"实质错误"：

| 排名 | 含义 |
|---|---|
| 0 | 精确匹配 |
| < 10 | 极接近 —— 前向基本正确，存在小误差 |
| < 100 | 接近 |
| < 1000 | 有一定相关性 |
| 更大 | 基本无关 —— 前向存在实质性错误 |

这是当前唯一能区分问题性质的判据。

### 4.3 关键澄清：`new_state-0` 是**最终状态**

第 11 轮解开的矛盾。`new_state-0` 是**整个 prompt（5 token）处理完后的状态**，
不是 token 0 的状态：

```
token 0: S = outer(k₀, β₀·v₀)                    （S 初值 0，delta 项为 0）
token 1: S = S·exp(g₁) + outer(k₁, β₁(v₁ − Sᵀk₁))  （delta 项非 0，S 已被衰减）
...
```

最终状态是**多次非平凡更新的叠加**，**不可能**再是 `outer(k, βv)` 形式。

**验证**：单步外积的行与行严格平行（比值恒为 `k[1]/k[0] = 0.9324`），
而官方 dump 的 `row1/row0 = [2.0 -1.5 -4.0]` 非常数 ——
**二者本来就不该相同**，因为官方是 5 步累积的结果。

> **所以 `new_state` 不能用来验证单步递推。** 要用它验证必须让 prompt 只有一个 token
> （此时最终状态才等于单步外积）；而单 token dump 里 `new_state` 打印为 `-0.0000`
> （`%12.4f` 精度不足，实际量级 ≈ 0.13 × 8.45 × 0.012 ≈ 1.3e-2，被截断）。

**下一步（具体可执行）**：
1. 让实现打印 token 0 之后的 `state[head0]`，与**单 token dump** 比对 —— 唯一干净的判据
2. 若不符，逐位核对 `beta`：从 `ssm_beta` 权重到 sigmoid 的整条计算

---

## 5. 工程状态

### 5.1 代码

| 文件 | 行数 | 状态 |
|---|---|---|
| `src/bonsai/gguf.c` / `gguf.h` | 428 / 286 | ✅ 41 项自测通过 |
| `src/bonsai/model.c` / `model.h` | 762 / 163 | 前向实现，6 阶段已验证 |
| `src/bonsai/pq2_0.h` | 228 | PQ2_0 编解码 + KNC IMCI 内核 + FWHT |
| `src/bonsai/test_gguf.c` / `test_forward.c` | 157 / 154 | 自测与逐阶段比对 |
| `scripts/bonsai_forward.py` | — | numpy 参考实现（oracle） |
| `scripts/bonsai_ref.py` | — | GGUF 加载器 + 编解码 + 自测 |
| `scripts/bonsai_parse_dump.py` | — | 官方 dump 解析器 |
| `scripts/bonsai_diff_gt.py` | — | 逐阶段 ground-truth 比对 |

### 5.2 工具链

| 需要 | 状态 |
|---|---|
| 宿主 C 编译器 | **Zig 0.17.0**（`D:\zig\...\zig.exe cc`，自带 clang）|
| 官方参考实现 | `D:\llama-prism-win\`（46 个工具，从 GitHub API 下载）|
| 模型 | `D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf`（7.21 GB，本地）|
| ground truth | `D:\bonsai\eval_callback_full.txt`（6.7 MB）、`dump_single.txt` |
| k1om 交叉编译器 | `icc.exe`（PSXE 2017）在 D 盘安装包内，**未安装** |
| 卡 | `mic0: online`，244 核 |

### 5.3 内存预算（16 GB 卡）

| 项 | 大小 |
|---|---|
| PQ2_0 权重（保持压缩态，流式解包） | 7.21 GB |
| KV cache（16 全注意力层，ctx 2048） | ≈ 0.067 GB |
| GDN 状态 + 卷积历史 | ≈ 0.05 GB |
| scratch | ≈ 0.13 GB |
| **合计** | **≈ 7.5 GB**，余量充足 |

### 5.4 预期速度

带宽上限：Mistral-7B Q8_0 实测 6.79 tok/s × 7.2 GB ≈ 49 GB/s 有效带宽。
`49 / 7.21 ≈ 6.8 tok/s` 理论值，扣除 PQ2_0 标量解码与 Hadamard 开销后
**预期 1-4 tok/s**。

---

## 6. 踩过的坑（每条都花了实际时间）

### 6.1 技术陷阱

1. **张量 `offset` 是相对数据区的**，不是绝对文件偏移。当绝对偏移会静默解出垃圾。
2. **`ftell()` 在 Windows x64 和 k1om 上都是 32 位 `long`**，对 7.2 GB 文件返回 -1，
   导致加载器静默失败。必须用 `_ftelli64` / `ftello`。
3. **`ssm_conv1d` 的真实布局是 `flat[c*ck + k]`**，与 `dims={4,10240}` 的字面顺序不符。
4. **embedding 的逆变换顺序是 `s ⊙ (H x)`**，与正向的 `H · (s ⊙ x)` 相反。
5. **`H` 含归一化因子，不要再除一次**（多除一次导致整体差 32 倍）。

### 6.2 dump 索引的坑（四次，已量化）

`common_debug_print_tensor` 对**每个轴独立**取「首 3 + 末 3」（`n=3`），`%12.4f` 打印：

1. 一行里的 6 个数字是**首 3 + 末 3**，不是「前 6 个」
2. 多 token 张量每个 token 占**一行**；取「整块最后 3 个」会拿到最后一个 token
3. 多 head 张量按 head 分组，每组 N 行；「最后 3 个数字」是**最后一个 head**
4. 4D 张量打印顺序是 `ne3 → ne2 → ne1 → ne0`（**反向**）

**规则**：先写出元素的**多维索引**，再映射到打印位置。
**不要**用「第 k 行」「最后 3 个」这类扁平化说法。

### 6.3 工程/环境陷阱

1. **PowerShell `Set-Content` 会破坏含非 ASCII 的文件**（见文首）。
   含中文的文件只能用 `write`/`edit` 工具修改。
2. **控制台代码页是 GBK**，中文输出在控制台显示为乱码 ——
   判断文件是否损坏要用 `read` 工具，**不能**凭控制台输出下结论。
3. **GitHub API 按 IP 限流**；`raw.githubusercontent.com` 与 git 协议被墙，
   但 `codeload.github.com` 可达，release 资产可用
   `Accept: application/octet-stream` 走 API 下载。
4. 本机无 gcc/clang/MSVC，WSL 无发行版；**Zig** 是唯一可用的宿主 C 编译器。

---

## 7. 参考

- 模型卡：[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)（`hf-mirror.com` 可直连）
- 发布说明：[PrismML — Introducing Bonsai 2 27B](https://prismml.com/news/bonsai-2-27b)
- 内核源码：[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) 分支 `prism`
  - `ggml/src/ggml-common.h` — `block_pq2_0` / `block_ptq1_0`
  - `ggml/src/ggml-quants.c:474-540` — `dequantize_row_pq2_0`
  - `ggml/src/ggml-cpu/ops.cpp:11015-11089` — GDN 递推
  - `ggml/src/ggml-cpu/ops.cpp:5944-6100` — RoPE 配对
  - `src/models/qwen35.cpp` — GDN/注意力层建图
  - `src/llama-model.cpp:1972-2139` — Hadamard 旋转构造
- 本地源码副本：`D:\llama-prism-src\llama.cpp-prism\`
