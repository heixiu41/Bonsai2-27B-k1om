# Codes-only repack（34 → 32 字节/块）：机制已证明，端到端无收益

**结论先说**：机制**成立且逐位一致**，内核在**单线程快 +7.8%…+9.2%**，
但在 244 线程的**端到端 decode 上效应落在噪声内**（交错同二进制 A/B 中位数 3.715 → 3.739 tok/s，**+0.66%**，
而噪声地板本身约 ±0.6%）。按本仓库的实证口径，它实现为 **`BONSAI_PACK=1` 可选，默认关闭**。

> 立项预估是 268 → ~238 ms/token（3.73 → ~4.1 tok/s）。**实测没有发生**，
> 原因见 §4：244 线程下这个 matvec 是内存延迟受限，不是指令受限。

---

## 1. 机制

PQ2_0 的一个块是 34 字节：2 字节 fp16 缩放 + 32 字节 2-bit 编码。
F3 内核每个块要两次 `PQ2_IMCI_LOAD16`（把 16 个**任意对齐**的字节扩展成 16 个 dword），
因为 code payload 在 `blk+2`，34 字节跨步让它**只 2 字节对齐**，只能走

```
vpxord        %zmm0, %zmm0, %zmm0
vloadunpackld (%rdi){uint8}, %zmm0
vloadunpackhd 64(%rdi){uint8}, %zmm0     <- 还多读 64 字节
```

drow 预计算落地后，权重流里的 fp16 缩放字节**已经是死字节**。把它丢掉，块跨步变成 32：

- 16 字节 code 片落在 `32b` / `32b+16`，**两者都是 16 字节对齐**；
- 于是可以用 KNC 的 **load-with-upconvert 寻址模式**，一条指令完成同一件事：

```
vmovdqa32 (%rdi){uint8}, %zmm0            <- 1 条，且不再越界读
```

`-S` 实测（`load_probe.c`）：

| | 指令 |
|---|---|
| `old_form`（PQ2_IMCI_LOAD16） | `vpxord` + `vloadunpackld` + `vloadunpackhd` = **3** |
| `new_form`（对齐 `extload_epi32(UINT8)`） | `vmovdqa32 (%rdi){uint8}` = **1** |
| `old_pair`（一个块的两半） | **6** |
| `new_pair` | **2** |

即每块从 ~6 条降到 2 条（去掉 `vpxord` 与寄存器搬运后，净省 ~2–4 条，占 F3 的 ~30 条的 7–13%），
与立项时估的 **~11%** 吻合。

## 2. 先证明：lane 映射（这是本任务的核心风险，已排除）

`PQ2_IMCI_LOAD16` 是 lo/hi **unpack** 的组合，而对齐形式是普通 upconvert **load**——
两者的 lane 顺序**没有理由相同**，若不同则 8 条 `permutevar` 查出的全是垃圾。

卡上实测（`src/probes/repack_probe.c`，`/tmp/repack_probe`）：

```
=== lane map on a ramp (expect dword i = p[i]) ===
     i     p[i]   LOAD16  aligned
     0        3        3        3
   ...
  mismatching lanes at offset 0: 0/16
=== every 16-byte-aligned offset in a 64-byte-aligned buffer ===
  16 offsets checked, 0 lane mismatches total
=== packed geometry: block stride 32, halves at 32b and 32b+16 ===
  packed offsets with mismatches: 0/12
VERDICT: aligned extload_epi32(UINT8) == PQ2_IMCI_LOAD16 bit-for-bit
```

在**全部 16 个 16 字节对齐偏移**上逐 lane 相同，packed 几何（含奇数块 `b=1,3,5`）也 0 失配。

## 3. 内核 A/B（`src/probes/pq2repack.c`，五种真实形状）

pq2ab 式方法：同进程双顺序交替、best-of-5、字节翻转防 CSE、volatile sink、
工作集 256 MiB（故意大于 244×512 KB 的聚合 L2）、1 与 244 线程，逐行逐位比对。

```
shape                       thr  34B G w/s  32B G w/s    ratio copies  bit-equal
5120x17408 (ffn gate/up)      1        2.0        2.2    1.079     11        yes
5120x17408 (ffn gate/up)    244      187.1      178.4    0.953     11        yes
17408x5120 (ffn down)         1        2.1        2.3    1.092     11        yes
17408x5120 (ffn down)       244      177.6      168.7    0.950     11        yes
5120x10240 (gdn qkv)          1        2.0        2.2    1.078     19        yes
5120x10240 (gdn qkv)        244      154.4      185.3    1.200     19        yes
5120x6144  (gdn z/value)      1        2.0        2.2    1.078     32        yes
5120x6144  (gdn z/value)    244      124.8      202.2    1.620     32        yes
5120x2048  (attn k/v)         1        2.0        2.2    1.080     96        yes
5120x2048  (attn k/v)       244      124.0      202.5    1.633     96        yes
```

- **逐位一致：五种形状 × 两种线程数，全部 yes。** 两个内核做的是**完全相同顺序**的 FMA，
  只有取数指令不同，所以这里要求的是**精确相等**而不是"接近"。
- **单线程：+7.8% … +9.2%，五种形状高度一致** —— 单线程是 issue-bound，指令数就是限制因素。
- **244 线程那列不可用**：比值 0.95–1.63，因为两个布局是**两次独立的大分配**，
  物理页放置不同；这一列测的是页放置，不是内核。真实数字只能靠端到端
  （原地重打包用同一批页，才是公平对照）。

## 4. 端到端：交错同二进制 A/B（决定性）

重打包是**原地且不可逆**的，所以不能在同一进程里来回切；但 `BONSAI_NO_PACK`
让同一个二进制保持 34 字节布局，即改动前的行为。**轮流交替**（而不是先跑完一组再跑另一组）
以抵消先前测到的 ±6% 跨进程漂移。每轮 64 token greedy：

| 轮 | 34B（no pack） | 32B packed | 配对胜者 |
|---|---|---|---|
| 1 | 3.794 | 3.853 | packed |
| 2 | 3.649 | 3.739 | packed |
| 3 | 3.715 | 3.705 | 34B |
| **中位数** | **3.715 tok/s** | **3.739 tok/s** | **+0.66%** |

六次运行的 top-1 全部是 **11751**。

### 怎么读这个数字

**+0.66% 与零不可区分。** 3 个配对里 2 个偏向 packed，但单次测量的散布是 ±1–2%，
而**噪声地板本身就约 ±0.6%** —— 这一点由本节当初的一次事故校准出来了（见 §4.1）。
所以正确结论是：**端到端效应落在噪声内**，而不是"repack 快 0.7%"，
更不是立项时预估的 **+11%**。

### 为什么没有收益

与 `K1OM_TIME_BUDGET.md` 的预判完全一致：**244 线程下这个 matvec 是内存延迟受限，
不是指令受限**，所以省掉 ~30 条里的 2–4 条不起作用（单线程的 +8% 只在 issue-bound 时体现）。
同理，读的字节少了 6% 也没帮助——说明该 regime 也不是带宽受限。

### 4.1 一个必须记下来的方法学陷阱：卡上二进制是旧的

本节第一版测出 **−0.62%**，我差点据此收工。真相是那次 A/B 的**两组配置跑的是同一个二进制**：

`scripts/chat.py::ensure_card_files()` 对卡上 chat 二进制**只判断"存在"，从不比对尺寸**
（SCIF sink 走 `_ensure_card_binary`，是带尺寸比对的；chat 二进制当时走的是另一条路径）。
于是本轮重编译出的 501465 / 501538 两版**从未上传**，卡上一直是上一轮 496950 的 engine-tweaks 版；
`BONSAI_NO_PACK` 当然毫无作用，两次测量只是同一配置的重复。

**"两个相同配置比出 −0.62%"顺便给出了噪声地板的实测值**，这反而成了判断 +0.66% 的标尺。

已修根因：`ensure_card_files()` 现在对 chat 二进制也走尺寸比对（`_card_file_size` vs
`os.path.getsize(local)`），不一致就重传。**任何卡上测速前先 `ls -l /tmp/bonsai_k1om_chat`
确认尺寸等于本地**。

## 5. 记忆体：立项时的"7.2GB→6.8GB"**不成立**

原地压缩把 code 往前挪 2 字节/块，留下**散布在同一个分配里的 2 字节空洞**——
没有新的分配，也没有任何一页可以被释放。实测两种布局的稳态 **VmRSS 都是 9.03 GiB**
（门禁 13.5 GiB，通过）。要真正省下常驻内存，得重建缓冲区（把各张量搬到新地址），
那会需要 open 期间的额外内存或第二次 mmap，与"禁止大额分配"的约束冲突。

## 6. 附带的一条真实改进：越界读消失

旧 F3 内核对每个块会**越界读到块后最多 48 字节**（pq2_0.h 顶部有实测记录，
gguf.c 因此把缓冲区多分配 64 字节并把尾巴清零）。packed 内核的加载正好覆盖它需要的
32 字节，**不再越界**。这是仅凭读指令就成立的事实，与速度无关。

## 7. 决策与开关

**默认关闭**（`BONSAI_PACK=1` 开启）。理由：端到端没有可复现的收益，而按仓库口径
"未确证的收益不默认开启"。机制本身已证明，开关保留给 issue-bound 的场景
（例如将来单线程或低线程数的变体）。

## 8. 验证门结果

| 门 | 结果 |
|---|---|
| 探针：五种形状 × {1,244} 线程逐位一致 | ✅ **0 失配**（§3） |
| 探针：G w/s 提升 | ✅ 单线程 **+7.8%…+9.2%**；244 线程不可测（§3） |
| 宿主回归 `test_forward`（zig cc） | 见 §9 |
| 卡上 `scripts/gate_repack.py` | ✅ top-1 = **11751 (' Paris')** MATCH；64 token **3.85 tok/s**（1.032x，单次测量，散布 ±1–2%）；`BONSAI_PACK=1` 六次运行 top-1 也全是 11751 |
| 内存：稳态常驻 ≤ 13.5 GB | ✅ **9.03 GiB** |

## 9. 文件

| 文件 | 作用 |
|---|---|
| `src/probes/repack_probe.c` | 证明 lane 映射逐位一致（§2） |
| `src/probes/pq2repack.c` | 五种真实形状的内核 A/B（§3） |
| `src/pq2_0.h` | `PQ2_0_PACKED_BLOCK_BYTES` / `PQ2_IMCI_LOAD16A` / `pq2_row_block_sum_packed` / `pq2_0_matvec_knc_xp_packed` / `pq2_repack_rows_inplace` |
| `src/model.c` | `mm_drows_for` 里在 drows 之后原地重打包并置 `t->packed`；`mm_run` 按 `dr && t->packed` 分发 |
| `src/gguf.h` | `bonsai_tensor.packed` |
| `scripts/gate_repack.py` | 卡上门禁（top-1 + 64 token 测速 + 内存） |

### 复现

```powershell
# 探针（卡端）
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT = 'C:\Program Files\Intel\MPSS'
& $icc -mmic -O2 -std=c11 -qopenmp -Isrc src\probes\pq2repack.c -o pq2repack
# 上传后： LD_LIBRARY_PATH=/tmp /tmp/pq2repack 256 5

# 门禁（默认不打包）
python scripts\gate_repack.py
# 打开打包
#   BONSAI_PACK=1 加入 chat 的远程命令即可
```
