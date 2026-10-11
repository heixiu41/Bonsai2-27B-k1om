# k1om 数据通道：SCIF RMA 直传模型（已落地）

> **本文件已更新。** 旧版本记录的是「消息式 SCIF（`scif_send`/`scif_recv`）只有
> 9–39 MB/s、GB/s 级通道拿不下模型」的状态。**那个问题已经解决**：现在是
> SCIF 注册内存 RMA 直传，实测 **357–359 MB/s**，7.2 GB 模型 **20 秒**传完，
> 比消息式快约 8 倍、比 scp 快约 28 倍。旧的分析与踩坑记录保留在文末。

---

## 1. 结论速查（全部为本轮真机实测）

| 通道 | 实测速率 | 状态 |
|---|---|---|
| **SCIF RMA 直传**（`scif_register` + `scif_writeto` + `scif_fence_signal`） | **357–359 MB/s** | ✅ **现在默认走这条** |
| SCIF RMA 裸链路带宽（主机→卡注册窗口） | **3.4–3.5 GB/s** | ✅ 与分片大小无关 |
| 消息式 SCIF（`scif_send`/`scif_recv`，16 MB 消息） | 42.5–42.8 MB/s | ⚠️ 保留为第二层回退 |
| scp（`chacha20-poly1305`） | 11.7–12.7 MB/s | ⚠️ 保留为第三层回退 |
| 卡端 tmpfs 写入（**端到端真正的瓶颈**） | **364–370 MB/s** | ❌ 硬顶，见 §3 |

7,206,168,928 字节实测：**20.1–20.3 s**，卡端 `flush 360 MB/s`，校验线程等待 `0.00 s`。

### 完整性（两道独立验证）

1. **传输内置全量校验和**：卡端对「写进文件的每一个字节」计算滚动 64 位校验和，
   主机对本地文件算同一个函数，逐次比对 —— `bytes OK, checksum OK`
   （本模型 `3ade1324e3a79a16`）。
2. **独立密码学校验**：卡上 **`md5sum`**（BusyBox 没有 `sha256sum`，只有 `md5sum` /
   `sha3sum`）与主机 `certutil -hashfile` 比对，7,206,168,928 字节全量一致：
   `e119192844abba9e7821f6b0e23022a3`。

---

## 2. 设计：256 MB 环形窗口 + fence 流控

卡只有 15.5 GB 内存、`/` 是 12.9 GiB 的 tmpfs，**不能整文件注册**，所以用环形窗口。
握手协议**完全不变**（约束 2）：主机仍然连 20500 端口，仍然先发 8 字节 LE 长度 +
248 字节 NUL 补齐文件名；只有数据面换成了 RMA。

```
主机                                           卡（k1om）
 │  scif_connect(20500)                          │
 │ ── 256 字节头（size + name，与旧 sink 逐字节相同）──▶
 │                                     scif_register(256 MiB 窗口)
 │ ◀── scif_rma_geom { rma_off, win, ctrl_off, slot_base, slot_size, nslots } ──
 │                                              │
 │ 循环 k = 0 … 总槽数-1：                        │
 │   fread → 注册好的 32 MiB 源缓冲                 │
 │   scif_writeto(async) 到 slot k%8             │
 │   scif_fence_mark                             │
 │   scif_fence_signal(SCIF_SIGNAL_REMOTE,       │
 │        ctrl[HOST_SEQ] = k+1)  ───────────────▶ 轮询 ctrl[HOST_SEQ] > k
 │                                              │   write() slot k → 文件偏移 k*slot
 │                                              │   ctrl[CARD_SEQ] = k+1
 │ ◀── scif_readfrom(ctrl[CARD_SEQ]) 做背压 ──────┘
 │     （永远不超前 8 个槽）
 │ ◀── scif_send(rma_result{status,bytes,ck,…}) ──
```

窗口布局（`ctrl_off = 0`，`slot_base = 4096`）：

```
+0        ctrl[HOST_SEQ]  主机已写完的槽数（fence_signal 远端写）
+8        ctrl[CARD_SEQ]  卡已落盘的槽数（主机用 scif_readfrom 读）
+16       ctrl[ABORT]
+4096 …   8 × 32 MiB 槽环
```

**关键点：控制区必须与槽区不重叠。** 第一版把控制区放在 `ctrl_off = 4096` 而槽 0 也从
`ctrl_off + 0` 开始，于是主机的 `HOST_SEQ` 写进了槽 0 的数据、而主机又从文件数据里读到
一个「看起来像槽数」的值（GGUF 的 tensor_count ≈ 851）当信用额度 —— 卡端
`stalled at slot 0 (host_seq=0)`，主机却一路跑到 89%。现在两端都校验
`ctrl_off + RMA_CTRL_BYTES <= slot_base`，主机还会拒绝 `card_seq > 总槽数` 的值。

### 为什么 fence 与 readfrom 都能用（本轮探针实测，不是凭记忆）

`tools/rma_probe_{sink,send}.c` 在真机上验证了 4 件在设计里必须成立的事：

| 断言 | 实测结果 |
|---|---|
| 主机 `scif_writeto` 的数据真的落到卡注册窗口 | 卡端校验和与主机**完全一致**，0 个坏字、0 个残留哨兵值 |
| `scif_fence_signal(SCIF_SIGNAL_REMOTE)` 的值能到卡内存 | 卡读出 `0xfeedfacecafebeef` ✅ |
| **卡端自旋能看见 RMA 写进来的控制字**（KNC L1 不保证一致，这是最大风险点） | `saw1=1 saw2=1`，两次增量都看见 ✅ |
| `scif_readfrom`（卡→主机） | 读出卡写下的 magic ✅ |

ABI 一律以 MPSS 3.8.6 头文件为准（约束 3）：卡端
`MPSS\k1om-mpss-linux\usr\include\scif.h`，主机端
`MPSS\sdk\include\scif.h`（`off_t` = `int64_t`、`scif_epd_t` = `HANDLE`）+
导入库 `MPSS\sdk\lib\lib_uSCIF.lib`，不再手写函数指针原型。

---

## 3. 为什么端到端是 357 MB/s 而不是 GB/s：卡端 tmpfs 硬顶

SCIF RMA 链路本身是 **3.4–3.5 GB/s**，但模型必须落到卡上，而卡上**唯一可写的文件系统
是 tmpfs**（`/tmp -> /var/tmp`，挂在 tmpfs，在卡的内存里）。`tools/rma_flushbench.c`
在卡上量出的数字：

| 卡端操作 | 实测 |
|---|---|
| `write()` 新页，1 / 4 / 16 / 32 / 64 MiB 分片 | 364 / 363 / 363 / 364 MB/s（**与分片无关**） |
| `pwrite()` 1 / 2 / 4 / 8 / 16 线程 | 364 / 365 / 364 / 364 / 363 MB/s（**完全不扩展**） |
| 重写同一文件（页已分配） | 575–584 MB/s |
| `posix_fallocate`（只分配、不拷数据） | 362 MB/s |
| 读回 | 621–642 MB/s |
| 纯 RAM `memcpy` | 2640 MB/s |
| 内置滚动校验和 | 765–837 MB/s（每 32 MiB 约 40 ms） |

**结论：瓶颈是 tmpfs 的页分配，而它在 2.6.38 内核里是全局串行的** ——
16 个线程和 1 个线程一样快；`posix_fallocate` 单独跑就是 362 MB/s（说明
「拷贝」几乎是免费的，代价全在分配页上），所以预分配是净亏（先花 362 MB/s 分配一遍，
再花 584 MB/s 写，合并只有 226 MB/s）。

因此端到端 **~370 MB/s 就是这台机器的天花板**，我们跑到 357–359 MB/s（约 96–97%）。

### 怎么把 223 MB/s 提到 357 MB/s

第一版卡端把校验和串在写盘之前，单核「校验 + 写盘」合并只有
`1/(1/366 + 1/837) ≈ 254 MB/s`，实测 223 MB/s。把校验和搬到**独立线程与写盘重叠**
（写槽 k 的同时校验槽 k；只有两者都完成才发布 `CARD_SEQ` 让主机复用该槽）之后，
卡端只剩写盘受限，`ck wait 0.00 s`，实测 357 MB/s。

> 注：多线程写盘没有意义（§3 表），所以没有做并行 flusher —— 数据不支持。

### 环深调参：8×32 vs 16×16 vs 4×64 vs 32×8（窗口都锁在 256 MiB）

`sink` 现在会累计**卡端等主机**的时间（`card-wait-host`：自旋等 `ctrl[HOST_SEQ]`），
这是判断环深够不够的直接指标；`tools/tune_rma_slots.py` 每档跑 2 次：

| 配置 | 窗口 | 端到端 | 卡端 flush | **card-wait-host** | host ring-wait | ck-wait |
|---|---|---|---|---|---|---|
| **8×32**（默认） | 256 MiB | **358.9 MB/s** | 360 MB/s | 0.045 s | 16.0 s | 0.000 s |
| 16×16 | 256 MiB | 356.6 MB/s | 357 MB/s | 0.020 s | 16.2 s | 0.000 s |
| 4×64 | 256 MiB | 355.9 MB/s | 357 MB/s | 0.080 s | 16.4 s | 0.000 s |
| 32×8 | 256 MiB | 357.0 MB/s | 357 MB/s | 0.010 s | 16.3 s | 0.000 s |

**结论（与预期一致：收益不大，因为卡端写盘已是硬顶）：**

1. **四档吞吐差 0.8%，全在噪声内**（8×32 两次跑出来 360.2 / 357.5）—— 环深对吞吐没有影响。
2. **卡端几乎从不因等主机而空转**：最差 0.080 s / 20 s = 0.4%，最好 0.010 s = 0.05%。
   连最浅的 4 槽环都够。fence/`HOST_SEQ` 的发布延迟**没有**成为瓶颈。
3. `card-wait-host` 的大小**按槽尺寸单调**（64 MiB→0.080 s，32→0.045，16→0.020，8→0.010），
   说明这点等待是**启动爬坡**（主机要先读满第一个槽才能让卡开工），不是稳态饥饿。
4. `host ring-wait ≈ 16 s / 20 s` 四档一致：**主机 80% 的时间在等卡**，卡 100% 是瓶颈 —— 与 §3 吻合。
5. `ck-wait = 0.000 s` 四档一致：重叠的校验线程在任何槽尺寸下都不会挡住 flusher。

所以**默认保持 8×32**：吞吐最高、启动爬坡只有 0.2%，同时每传一次只有 215 次 fence
（32×8 要 900 次）。

---

## 4. 文件与用法

| 文件 | 作用 |
|---|---|
| `tools/scif_rma.h` | 两端共享的协议常量、窗口几何、校验和 |
| `tools/scif_rma_sink.c` → `tools/scif_rma_sink` | 卡端 sink（`icc -mmic`，EM_K1OM 0xB5） |
| `tools/scif_rma_send.c` → `tools/scif_rma_send.exe` | 主机端 sender（`zig cc` + `lib_uSCIF.lib`） |
| `tools/build_rma.ps1` | 一次性构建两者，并校验 ELF `e_machine` |
| `tools/tune_rma_slots.py` | 环深调参（上表就是它跑出来的） |
| `tools/rma_probe_sink.c` / `rma_probe_send.c` | 可行性探针（§2 那张表的来源） |
| `tools/rma_flushbench.c` | 卡端写盘/读回/校验和基准（§3 那张表的来源） |
| `scripts/verify_transfer_layers.py` | 逐层验证三层回退 + md5 开关正/负向测试 |
| `tools/scif_send.c` / `scif_sink.c` | **原样保留**，作为第二层回退，未改动 |

**入库策略**（跟着仓库既有先例走）：`tools/scif_sink` 这个**卡端**二进制本来就被跟踪，
所以同样跟踪 `tools/scif_rma_sink` —— 一个干净 checkout 不需要 icc 就能跑第一层。
诊断用二进制（`rma_probe_sink`、`rma_flushbench`）已在 `.gitignore` 里显式排除
（`build_rma.ps1 -Probe` 可重建），主机侧 `.exe`/`.pdb` 沿用仓库原有的 `*.exe`/`*.pdb` 规则。

```powershell
# 构建（PowerShell 5.1 的默认执行策略会拦 .ps1，必须带 Bypass）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build_rma.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build_rma.ps1 -Probe   # 连探针+基准一起

# 环深调参（每档 2 次，约 5 分钟）
python tools\tune_rma_slots.py --runs 2
python tools\tune_rma_slots.py --configs 8x32,4x64 --runs 3

# 手工跑一次传输
ssh -i $env:USERPROFILE\.ssh\id_rsa_mic root@192.168.1.100 "export LD_LIBRARY_PATH=/tmp; /tmp/scif_rma_sink"
.\tools\scif_rma_send.exe D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf bonsai.gguf
```

卡端可调：环境变量 `SCIF_RMA_SLOT_MB`（默认 32）、`SCIF_RMA_SLOTS`（默认 8），
窗口 = `4096 + nslots*slot_size`，默认 256 MiB。主机端并发源缓冲 4 个（`NBUF`）。

### 三层回退与验证

`scripts/chat.py::ensure_card_files()` 现在按 **RMA → 消息式 SCIF → scp** 依次尝试，
`BONSAI_TRANSFER_LAYER=auto|rma|scif|scp` 可强制某一层（强制时禁用回退）。

```powershell
# 每一层单独跑通（64 MiB 切片，md5 比对）
python scripts\verify_transfer_layers.py --mb 64
#   实测: rma 18.7 MB/s / scif 14.0 MB/s / scp 11.7 MB/s （小文件，启动开销占比大）
#   同一个 64 MiB 切片上 RMA 是 307 MB/s、消息式 42.5 MB/s（见 sender 自身输出）

# 自动回退链：藏掉第一层 sender，必须落到第二层
python scripts\verify_transfer_layers.py --mb 64 --fallback
# 自动回退链：两层 SCIF 都藏掉，必须落到 scp
python scripts\verify_transfer_layers.py --mb 64 --fallback scif

# 可选整文件 md5 复核开关的正/负向测试
python scripts\verify_transfer_layers.py --mb 64 --md5
```

三层均已实测 PASS，且落卡文件的 md5 与本地一致。

### 传输后可选整文件 md5 复核

传输**本身**已经端到端比对了一个覆盖全部字节的滚动校验和（§1），所以默认不再重算哈希。
需要「密码学级别」的二次确认时（怀疑卡在抽风、或想留证据）：

```powershell
set BONSAI_VERIFY_MD5=1
python scripts\chat.py --verify-md5        # 等价，命令行开关更好记
```

它会在上传完成后，主机流式算一遍 md5、卡上跑 `md5sum`，不一致就返回失败。

- **卡上没有 `sha256sum`**（BusyBox 只编译了 `md5sum` 和 `sha3sum`），所以只能用 md5。
- **很慢**：主机 7.2 GB 约 14 s，卡上 `md5sum` 约 **130 s**（1.2 GHz 标量）——
  这是它默认关闭的原因。`--md5` 自测（`verify_transfer_layers.py --md5`）用同尺寸随机数据
  覆盖卡上副本，确认**只比对内容、不比对尺寸**也能检出。


### 冷启动端到端

```
python scripts/chat.py            # 冷卡（tmpfs 里没有模型/没有 libiomp5.so）
```
实测 **exit 0 / 60.6 s**：RMA 上传 7.2 GB（359.3 MB/s，20.1 s）→ 卡加载模型 16.4 s →
READY 24.1 s → 预填充 15 tok / 5.9 s → 生成 12 tok / 3.4 s（3.55 tok/s）→ `/quit`。

---

## 5. 本轮修掉的四个坑（都会影响「冷启动能不能跑通」）

1. **窗口控制区与槽 0 重叠**（见 §2）——信号全丢，表现为卡端 `stalled at slot 0`。
2. **`ssh` 子进程继承父进程 stdin，把用户输入整个吃掉。** `chat.py` 用
   `subprocess.Popen(["ssh", …])` 起 sink 时没设 `stdin`，`ssh` 会把 stdin 转发给远端命令，
   于是管道喂进去的问题被吞掉，冷启动表现为「打印完 `You> ` 立刻 EOF 退出、退出码还是 0」。
   现在 `chat.py` 统一用 `_run_detached()`（默认 `stdin=DEVNULL`），
   `chat_server.py` 的两处 ssh 也补上了。
   **旧的消息式 SCIF 路径同样有这个缺陷**，只是以前很难撞上。
3. **`libiomp5.so` 从来没被部署过。** 卡端 `bonsai_k1om_chat` 是 OpenMP 构建，
   `ensure_card_files()` 只传了二进制和模型；卡一重启（tmpfs 清空）就
   `error while loading shared libraries: libiomp5.so`。现在会自动从
   `…\intel64_win_mic\libiomp5.so` 推上去（可用 `BONSAI_MIC_OMP_LIB` 覆盖）。

4. **`ensure_card_files()` 里的尺寸校验一直是坏的 —— 而且是两重坏。**
   原代码用 `stat -c %s`，但**卡上根本没有 `stat`**，所以永远静默失败、永远走 scp 回退。
   换成 `wc -c` 也是错的：**这块卡上的 BusyBox `wc -c` 是 32 位的**，把
   7,206,168,928 报成 **2,911,201,632**（正好是 `size mod 2^32`）。
   现在解析 `ls -l` 的第 5 个字段拿真正的 64 位大小
   （`_card_file_size()`，实测返回 7206168928 与主机一致）。

   > 这个坑对 `_ensure_card_binary()` 的影响是隐蔽的：卡上二进制只有几十 KB，
   > 32 位够用，所以不会暴露；但只要哪天要按大小比对 >4 GiB 的东西就会静默出错。

### 附：卡上工具的两个额外陷阱（写验证脚本时踩到）

- **BusyBox `dd` 没有 `conv=notrunc`**（`dd` 的 usage 只有 `if/of/bs/count/skip/seek`）。
  想"原地改一个字节"做负向测试会静默失败 —— 命令报 usage，文件一个字节都没动，
  于是"注入损坏"根本没注入，测试还会假通过。要改内容就**整文件重写**
  （`dd if=/dev/urandom of=... bs=1M count=N`），顺便还能证明校验比的是内容而不是尺寸。
- **卡上 `getconf`、`stat` 都不存在**；`wc -c` 是 32 位的（见上一节）。
  能信的只有 `ls -l` / `wc` / `head` / `dd` / `killall` / `pidof` / `ps` / `md5sum` / `sha3sum`。

---

## 6. 历史记录（旧版本内容，保留）

### 旧实测数据

| 通道 | 速率 | 当时用了吗 |
|---|---|---|
| SCIF DMA / COD offload 引擎 | 理论 3–5 GB/s | ❌ |
| SCIF read/write（`micnativeloadex` 走这条） | 1–2 GB/s | ✅ 已打通 |
| mmap BAR + memcpy (PIO) | 几百 MB/s | ❌ |
| **SSH 虚拟以太网** | **9.0–15.7 MB/s** | ⚠️ 之前一直用它 |

SSH 实测：256 MB / 28.6 秒 = **9.0 MB/s**；6.71 GB 模型 = **7.7 分钟 / 14.8 MB/s**。
本轮复测：scp 11.7–12.7 MB/s，与历史一致。

**根因**：KNC **没有 AES-NI**（`/proc/cpuinfo` 里连 `sse2` 都没有，它是独立 ISA），
SSH 加密全靠 1.2 GHz 单核软件实现。换 `chacha20-poly1305` 只能快 1.6x ——
**在最慢的通道上做微调，方向本身就错了。**

### `micnativeloadex` 配方（走 SCIF，不经 SSH）

```powershell
$ml  = 'C:\Program Files\Intel\MPSS\bin\micnativeloadex.exe'
$sys = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux'
$omp = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\compiler\lib\intel64_win_mic'

# 注意变量名是 SINK_LD_LIBRARY_PATH（不是 LD_LIBRARY_PATH）
$env:SINK_LD_LIBRARY_PATH = "$sys\lib64;$sys\usr\lib64;$omp"
& $ml D:\zigtest\k1om\bonsai_k1om -a "/tmp/bonsai.gguf selftest" -t 900
```

踩过的坑：变量名是 `SINK_LD_LIBRARY_PATH`；它在**宿主侧**解析依赖，要指向 k1om sysroot；
缺 `libiomp5.so` 时要加 `intel64_win_mic`；它只接受自己的路径做执行，不能代理卡上读宿主文件。

### 当时仍未解决、现在已解决的事

- ~~怎么让 GB/s 级通道覆盖模型文件~~ → §2 / §3：RMA 直传，**357–359 MB/s**。
- COI / `mic0.filelist` / `micctrl --addcifs` 都不需要了。
