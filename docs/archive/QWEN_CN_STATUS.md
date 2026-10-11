# Qwen2.5 中文适配 & Bug 修复状态

## 已完成修改

### 1. 中文 UTF-8 显示 (`src/run.c`, `safe_printf`)

**问题**: `safe_printf()` 过滤掉 `>= 0x80` 的单字节，导致中文 UTF-8 多字节序列的原始字节被丢弃，chat 模式下中文 token 无法显示。

**修复**: 允许 `byte_val >= 0x80` 通过。这些是 Qwen tiktoken tokenizer 的 byte-fallback 产生的原始 UTF-8 字节。

### 2. NLP 新参数 (`-e`, `-c`)

| 参数 | 用途 | Qwen2.5 使用值 |
|------|------|---------------|
| `-r 1000000` | RoPE theta | Qwen2.5 用 1e6，默认 1e4 |
| `-b 0` | 禁用 BOS token | Qwen 的 BOS 不是 id=1 |
| `-e 151645` | **EOS token ID** (新增) | Qwen 用 `<\|im_end\|>` = 151645 |
| `-c qwen2` | **Chat 模板** (新增) | `<\|im_start\|>...<\|im_end\|>` 格式 |

原有 Llama2 默认值不变。

### 3. Q/K/V Projection Bias（核心 bug 修复）

**问题**: Qwen2.5 的 Attention 模块有 `q_proj.bias`、`k_proj.bias`、`v_proj.bias` 共 84 个 bias tensor。原项目（llama2.c 派生）完全不处理 bias，因为 Llama2 没有。这导致 layer 0 attention 输出与 HF 参考偏差 ~5.97（L2 norm），是之前输出乱码的根因。

**修改**:
- `scripts/export_qwen.py`: 导出 q/k/v bias（FP32），header 加 `has_bias=1` flag
- `src/run.c` TransformerWeights: 新增 `q_bias/k_bias/v_bias` 成员
- `src/run.c` read_checkpoint: 检测 `has_bias` flag
- `src/run.c` memory_map_weights_q80: 映射 bias 数据
- `src/run.c` forward_mic_q80: 应用 bias

### 4. Qwen2.5 Chat 模板

```text
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
```

### 5. Encode/Decode 适配

- `encode()` EOS token 从硬编码 2 改为使用全局 `eos_id`
- `decode()` BOS 空格剥离只在 `add_bos=1` 时执行
- `generate()` 和 `chat()` 中所有 `token == 2` 检查改为 `token == eos_id`

---

## 测试结果

### Generate 模式（中文 prompt）

```
输入: 很久以前，在一个美丽的森林里，住着一只小兔子。
输出: 这里是一为的系的+也行后。以那里。仍然什么的一个小取公司生活的。
      舞习的基本贵也的的一里面的...
```

首 token 预测与 HF transformers 参考一致（token 3837 `，`），中文可辨识，但仍有少量杂质。127 tokens, 16.2 tok/s。

### Chat 模式（Qwen2 模板）

Chat 模式加载正常，但因 attention 路径残差，多轮对话质量待评估。

### Tok/s

约 16-17 tok/s（1.5B Q8_0, 244 threads, Xeon Phi 7120P）。

---

## 已知残留问题

### 1. ~~KNC 上 k_bias 导致 NaN~~ ✅ 已修复（2026-08-02）

**现象**: 在 forward_mic_q80 中 `s_k[i] += k_bias[...]` 会导致所有 logits 变成 NaN。

**排除过的可能原因**（均不是根因）:
- 数据损坏：Python 验证 k_bias 值干净，无 NaN/Inf
- 文件偏移错误：Python 和 C 代码计算的偏移完全一致
- mmap 内存访问问题：堆拷贝也崩
- OpenMP 并发问题：OMP_NUM_THREADS=1 也崩
- 编译器优化：-O0 也崩
- 浮点精度设置：-fp-model precise 也崩
- 循环结构：volatile 读、memcpy、4路展开都试过

**已知事实**:
- 只加 k_bias 前 3 个元素（值 4.1/-0.5/-1.8）→ 正常
- 加 128+ 个元素 → NaN
- k_bias 元素值范围 -316 到 +288（比 q_bias 的 -26~+29 大 10 倍）

**真正的根因**（两处,均为 run.c 的 MIC softmax 缺陷,与 KNC 硅层精度无关）:

1. **max_val 读到未初始化内存**（NaN 的直接来源）
   `softmax()` 里 `__m512 max_val_load = _mm512_load_ps(x)` 一次读 16 个元素,即使 `size < 16`
   （pos=0..14 时 size=pos+1）。`x[size..15]` 是上一次 softmax 留下的垃圾值;一旦含 NaN,
   `_mm512_reduce_max_ps` 得到 NaN,`expf(x[0]-NaN)=NaN`,整层输出变 NaN。
   k_bias 使各 head 的 score 值域变化,恰好把垃圾里的 NaN 触发出来（与「加 128+ 元素才崩」吻合——
   前 3 个 bias 时 score 变化小,垃圾仍是旧的非 NaN 值）。
   **修复**: max_val 只在有效 `size` 元素内计算;`size >= 16` 才走向量化 reduce,否则走标量。

2. **vexp2ps 对超大负数输入的越界 NaN**（pos≥16 时触发）
   应用 k_bias 后 attention score 值域达 ±324（无 k_bias 仅 ±9.4）,`exp(x-max)` 的输入到 -324,
   乘 log2e 后约 2^-467,远低于 KNC 硬件 `vexp2ps` 的次正规下界(≈2^-149),其「整部加到指数字段」
   的范围归约产生 NaN 而非干净的 0。
   **修复**: `_mm512_exp_ps` 的输入钳位到 `[-80, 0]`（低于 e^-80 的项归一化后本来就≈0）。

**当前状态**: k_bias 已去掉 `#ifndef __MIC__` 条件编译,**KNC 上正常应用 q/k/v 全部 bias**。
FFN sigmoid 的 `_mm512_exp_ps` 也加了 `[-88, 88]` 钳位（同根因防御）。验证:
- DBG_NAN 全链路 0 个 NaN;layer0 s_q/s_k/s_v 值干净
- 首 token 仍为 3837（，）与 HF 参考一致
- 中文生成连贯,~18 tok/s（244 threads, Q8_0）

### 2. 程序退出段错误 (Segmentation Fault)

每次运行结束时出现 `sh: line 1: NNNNN Segmentation fault`。怀疑是 free/munmap 清理阶段的问题，不影响推理结果，但需排查。

### 3. ~~RMSNorm eps 差异~~ ✅ 已修复（2026-08-02）

代码 rmsnorm 硬编码 `ss += 1e-5f`（Llama2 的值），而 Qwen2.5 的 `rms_norm_eps=1e-6`。

**为何差异显著**：Qwen2.5 的 token embedding 幅值很小，layer0 的 `mean(x²) ≈ 0.0005~0.001`，
eps 从 1e-6 换成 1e-5 会让方差相对变化 **~1.8%**、rsqrt 偏移 ~0.9%，layer0 输出 L2 diff 达
**0.11~0.26**（实测 token 52801 为 0.114，与 DEBUG_LOG 记录完全吻合），并经 28 层累积放大。

**修复**：
- `src/run.c` rmsnorm 改用全局 `rms_eps`（默认 1e-5，Llama2 兼容）
- 新增 `-E <eps>` 命令行参数
- 读 checkpoint 时检测 `has_bias=1`（Qwen 特征）自动切换 `rms_eps=1e-6`，显式 `-E` 优先

**验证**：Qwen 自动打印 `rms_eps=1e-06`；stories110M（无 bias）保持 1e-5，输出不受影响。
首 token logits 3837 从 12.6158 → 12.6256（均在 Q8_0 噪声内）。1 线程 greedy 下该 prompt 两值输出一致
（微小 logit 差不足以翻转 argmax）；采样时概率分布有细微偏移。

> 附注：发现一个独立的**并行非确定性**现象——`OMP_NUM_THREADS=244` 时相同命令两次 greedy 输出不同
> （OpenMP 并行归约浮点求和顺序差异 → 微小 logit 噪声 → 长序列路径分叉）。1 线程下输出确定。
> 这独立于 eps，不影响 eps=1e-6 的正确性结论。

### 4. ~~Chat 模式输出乱码~~ ✅ 已修复（2026-08-02）

三个独立问题：

1. **特殊 token 编码错误（核心）**：Qwen 的 `<|im_start|>`/`<|im_end|>` 是 tiktoken
   "added tokens"，靠正则整体匹配；而 run.c 的 encode() 只有字节级+merge 路径，中间合并串
   （如 `"<|"`）不在词表，无法重建 → 被拆成 `<`(27) `|`(91) `im`(318)... 模型看到完全错误的输入。
   **修复**：encode() 增加特殊 token 整体匹配（词表查 id，不硬编码；已与 HF transformers 逐 token 验证一致：
   `[151644, 872, 198, ...]` 完全吻合）。

2. **prompt 内的 EOS 触发回合切换**：Qwen chat 模板在用户消息后内嵌 `<|im_end|>`(==eos_id)，
   原代码 `if (token == eos_id) user_turn=1` 对 prompt token 也生效 → 喂 prompt 中途被打断。
   **修复**：仅模型采样的 EOS（`!from_prompt && token == eos_id`）才切换回合。

3. **stdin 提示符未 flush**：`read_stdin()` 的 `printf(guide)` 无 fflush，管道/ssh 下 stdout
   全缓冲 → 提示符永远到不了对端，交互死锁。**修复**：printf 后加 `fflush(stdout)`。

**验证**：chat 多轮对话正常（`printf '\n问题\n' | ./llama2_k1om ... -m chat`），
首轮回复连贯（如 "当然，我很高兴为你解答..."）；质量上限仍是 Q8_0 量化本身的退化。

### 5. 对话窗口脚本（scripts/chat_window.py）

Windows 侧交互式聊天窗口：ssh 管道驱动 `-m chat`，流式显示，多轮对话，支持 exit/quit/Ctrl+C。

踩过的两个 Windows 管道坑：
- `bufsize=0` 无缓冲管道下 `read(4096)` 会假性返回 `b""`（进程还活着）→ 必须 `read(1)` 逐字节
- 脚本内 `os.system("chcp 65001")` 会破坏后续 `input()`（重定向 stdin 时立即 EOF）→ 用
  `ctypes.windll.kernel32.SetConsoleCP/SetConsoleOutputCP(65001)` 替代

用法：`python scripts/chat_window.py [--threads 244] [--system "..."]`

### 6. 输出仍有少量杂质

中文输出有时夹杂英文/日文/符号。可能原因：
- Q8_0 量化损失（~0.4% 误差在 28 层中累积）——主要嫌疑
- 小模型（1.5B）固有的长序列重复/退化
- softmax 对 pos<16 的历史 max_val bug 会影响生成开头几个 token（已修复,但若改用旧二进制则仍存在）

### 7. Seq_len 上限 2048

KNC 16GB GDDR5 限制，Qwen2.5 的默认 max_position_embeddings=32768 在 run.c 中被截断到 2048。

---

## 二进制文件格式

```
Offset    Size       Content
0         28         Config header (7×int32)
28        1          Q8_0 flag (=1)
29        4          Group size (=64)
33        4          shared_weights flag
37        1          has_bias flag (=1)        ← NEW
38        26         Padding to 64 bytes
64        ...        Token embedding (Q8_0)
        ...          RMS attn weights (FP32)
        ...          Wq (Q8_0 per-layer interleaved)
        ...          Q bias (FP32 per-layer)    ← NEW
        ...          Wk (Q8_0)
        ...          K bias (FP32 per-layer)    ← NEW
        ...          Wv (Q8_0)
        ...          V bias (FP32 per-layer)    ← NEW
        ...          Wo (Q8_0)
        ...          RMS ffn weights (FP32)
        ...          W1/W2/W3 (Q8_0)
        ...          RMS final weight (FP32)
        ...          Freq_cis (FP32 cos+sin)
        ...          Wcls (Q8_0, or shared with embedding)
```

向后兼容：旧 `.bin` 文件 offset 37 为 0（padding），被读取为 `has_bias=0`，bias 指针设为 NULL，行为不变。

---

## 使用示例

```bash
# Qwen2.5 generate 模式
NO_DUMMY=1 ./llama2_k1om qwen15b_q80.bin \
  -z tokenizer_qwen.bin \
  -r 1000000 -b 0 -e 151645 \
  -i "很久以前，在一个美丽的森林里" -n 256

# Qwen2.5 chat 模式
NO_DUMMY=1 ./llama2_k1om qwen15b_q80.bin \
  -z tokenizer_qwen.bin \
  -r 1000000 -b 0 -e 151645 -c qwen2 \
  -m chat -N 4
```

---

## 文件清单

| 文件 | 状态 |
|------|------|
| `src/run.c` | 已修改（safe_printf, bias, new flags, chat template） |
| `scripts/export_qwen.py` | 已修改（bias export, has_bias flag） |
| `scripts/export_qwen_tokenizer.py` | 未修改（tokenizer 原已验证正确） |
| `scripts/verify_qwen.py` | 未修改（验证用） |
| `models/qwen15b_q80.bin` | 已重新导出（含 bias, 1.66 GB） |
| `models/tokenizer_qwen.bin` | 不变（2.1 MB） |
| `docs/QWEN_KNC_DEBUG_SUMMARY.md` | 之前的手动调试记录 |
| `docs/QWEN_CN_STATUS.md` | 本文档 |
