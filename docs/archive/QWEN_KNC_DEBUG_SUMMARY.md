# Qwen2.5-1.5B 在 KNC（Xeon Phi）上的中文生成调试流程总结

> 目标：让 Qwen2.5-1.5B-Instruct 能在 Intel Xeon Phi（k1om）上流畅输出连贯中文。
> 环境：KNC 独立 Linux（192.168.1.100，root，密钥 `~/.ssh/id_rsa_mic`），用 `icc.exe` k1om 交叉编译，运行时 `export LD_LIBRARY_PATH=/tmp`（加载 `libiomp5.so`），内存受限（`seq_len` 上限 2048）。

---

## 1. 背景

1. 用户先测试 `mistral7b_q80.bin` → 中文输出差（控制标记乱码）。结论：Mistral-7B 本身不擅长中文。
2. 改用 **Qwen2.5-1.5B-Instruct**（更擅长中文），模型由用户手动下载到 `models\qwen2.5-1.5b\`。

## 2. 已完成的工作链

| 阶段 | 产出/动作 | 状态 |
|---|---|---|
| tokenizer 导出 | `scripts\export_qwen_tokenizer.py` → `models\tokenizer_qwen.bin`（2192026B） | ✅ 已验证 |
| 模型导出 | `scripts\export_qwen.py --q80` → `models\qwen15b_q80.bin`（1657169984B） | ✅ 已验证 |
| 校验脚本 | `scripts\verify_qwen.py`（tokenizer 结构+roundtrip）、`scripts\check_q80_export.py`（反量化 vs 原版，误差~0.4%） | ✅ 通过 |
| run.c 适配 | `-r`(rope_theta)、`-b`(add_bos)、`-i`/`-z` 参数；Q8_0 读取 sw_flag；段错误修复 | ✅ |
| 部署 | scp -O 上传，`/tmp/llama2_k1om_*`、`/tmp/qwen15b_q80.bin`、`/tmp/tokenizer_qwen.bin` | ✅ |

### 2.1 关键文件格式（Q8_0，与 run.c `memory_map_weights_q80` 对齐）

- Header：7×int32（dim/hidden/layers/heads/kv/vocab/seq），flag=1，GS=64，**sw_flag（offset 33，tied embedding=1）**，填充到 64B。
- 权重布局：**逐层交错** `[L0 int8][L0 scales][L1 int8][L1 scales]...`（每层 int8 + 每 64 元素一个 FP32 scale）。非批量。
- `tie_word_embeddings=true` → vocab_size 为正、wcls 复用 embedding 表。

## 3. 修过的 Bug（按时间）

1. **tokenizer 编码**：utf-8 直存导致中文 lookup 失败 → 改为 tiktoken `bytes_to_unicode` 反映射（latin-1 字节）。
2. **vocab 缺失 293 个 token**：合并 `added_tokens`（<|im_start|> 等 id 151643–151664）+ 按 target_vocab 补齐。
3. **`fread(len=0)` 崩溃**：padding token 从 `b""` 改为 `b"\x00"`。
4. **权重布局错误（核心）**：导出为「全层 int8 + 全层 scales」，但 run.c 按逐层交错读取 → 输出退化为重复字符。改为逐层交错后输出改善。
5. **结束段错误（exit 139）**：Q8_0 模式跳过 "Mallocing on MIC" 却执行 "Freeing on MIC" → 给 4 处 offload 分配/释放块加 `&& !w->use_q80` 保护。
6. **RoPE in-place bug（本次会话修复）**：C 实现 `q[i+1] = q[i]*fci + q[i+1]*fcr` 用到被覆盖的 `q[i]`（应为原值）。已将 run.c 中 **10 处（5 个 forward 实现 × 2 段循环）全部改为临时变量版**，重新编译为 `build\llama2_k1om_rope` 并上传。
7. **dummy 前缀（本次会话发现）**：run.c `encode()` 会给 prompt 前加空格 token（llama2.c 遗留）。Qwen 不需要。已加 `NO_DUMMY=1` 环境变量开关，启用后首步 logits 与 numpy 参考一致。

## 4. 诊断方法（本次会话核心）

建立三层对照基准，逐层定位：

```
HuggingFace transformers（官方实现，FP32）= 基准真值
        ↕ 对比
numpy 参考前向（scripts\ref_forward.py，纯 FP32 手写 forward）
        ↕ 对比
KNC run.c（Q8_0 + 量化权重 + MIC forward）
```

### 4.1 关键实验与结论

1. **transformers 真值**（`scripts\ref_hf_golden.py`）：「好」= 单个 token **[52801]**，首步 top-1='，'(3837)（logit 12.59）。
   - 反证了我之前 numpy 的 bpe 编码 [161,98,121] 是**错的**（merge 查找用字节 vs 字符串键，永不合并）。tokenizer 链路最终以 run.c 式 greedy encode 验证正确：'好'→[52801] 与 transformers 一致。
2. **RoPE 判定**：pos=0 时 RoPE 为恒等（cos=1,sin=0），**单 token 对比测不出 RoPE bug**。此前「inplace 产生乱码」结论输入就是错的，作废。
3. **C vs numpy 首步 logits**（NO_DUMMY=1 后）：C 输出 top-1=','(11) logit 15.54，numpy 也 16.12，**两者一致**；但 transformers 为 '，'(3837)=12.59。→ 修复 RoPE + 关闭 dummy 后，C forward 与 numpy 参考一致，但 numpy/C 与 transformers 仍有系统偏差（ASCII 低 id token 偏高、中文高 id token 偏低）。
4. **逐层对比（本次最新发现）**：embed diff=0.000000（完全一致）；layer0 rmsnorm 输出 diff=0.114（疑似 rms_norm_eps：numpy 用 1e-5，Qwen 用 1e-6）；**layer0 attention 输出 diff=5.97（线性传播不可能放大这么多 → attention 路径存在真 bug）**；后续层全面发散（diff≈10~1700）。

## 5. 当前结论

- tokenizer、模型导出、权重布局：✅ 已验证正确。
- run.c 的 RoPE in-place bug：✅ 已修复并部署。
- dummy 前缀：✅ 已加开关（NO_DUMMY=1）。
- **仍存在未定位 bug**：layer0 的 attention 输出与 HF 参考相差巨大（diff=5.97），q/k/v 投影与头部拼装、或 attention 数值路径是当前最大嫌疑。
- 附带问题：KNC 运行结束时仍出现 Segmentation fault（与之前的段错误可能不同位置，待查）。

## 6. 下一步（待办）

1. 用 q_proj/k_proj/v_proj 子模块 hook 对比 layer0 的原始 q/k/v（已完成 hook 与 numpy dump 代码，未跑）。
2. 若 q/k/v 一致 → 检查注意力头部拼装（GQA，kv_mul=6）与 `attn_out = v_assembled @ wo.T`。
3. 若 q/k/v 不一致 → 检查 q/k/v 权重读取顺序/维度。
4. 修复后重测：`ssh ... "/tmp/llama2_k1om_rope /tmp/qwen15b_q80.bin -z /tmp/tokenizer_qwen.bin -r 1000000 -b 0 -n 80 -i '很久以前，在一个美丽的森林里，住着一只小兔子。'"`（预期输出连贯中文）。
5. 处理运行结束段错误。

## 7. 常用命令速查

```powershell
# 交叉编译（k1om）
powershell -ExecutionPolicy Bypass -File .\scripts\build_k1om.ps1 -Out llama2_k1om_rope -Define DBG_LOGITS

# 上传与运行
scp -O -i $env:USERPROFILE\.ssh\id_rsa_mic build\llama2_k1om_rope root@192.168.1.100:/tmp/
ssh -i $env:USERPROFILE\.ssh\id_rsa_mic root@192.168.1.100 `
  "cd /tmp && DBG_LOGITS=1 NO_DUMMY=1 LD_LIBRARY_PATH=/tmp OMP_NUM_THREADS=244 ./llama2_k1om_rope /tmp/qwen15b_q80.bin -z /tmp/tokenizer_qwen.bin -r 1000000 -b 0 -n 80 -i '好' > /tmp/out.txt 2>&1; cat /tmp/out.txt"

# 参考前向对照
python scripts\ref_hf_golden.py   # 生成 HF 真值 _hf_*.npy
$env:COLLECT="1"; python scripts\ref_forward.py   # numpy 逐层对比
```
