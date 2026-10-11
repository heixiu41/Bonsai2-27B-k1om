# k1om 数据通道：SCIF 才是快路，SSH 是最慢的一条

## 实测数据（本轮全部实测，非估算）

| 通道 | 速率 | 我用了吗 |
|---|---|---|
| **SCIF DMA / COD offload 引擎** | 理论 3–5 GB/s | ❌ |
| **SCIF read/write**（`micnativeloadex` 走这条） | 1–2 GB/s | ✅ 已打通 |
| mmap BAR + memcpy (PIO) | 几百 MB/s | ❌ |
| **SSH 虚拟以太网** | **9.0–15.7 MB/s** | ⚠️ 之前一直用它 |

SSH 实测：256 MB / 28.6 秒 = **9.0 MB/s**；6.71 GB 模型 = **7.7 分钟 / 14.8 MB/s**。

**这与历史记录一致**：`DEBUG_LOG.md` 12.10 节写着
`100% 595KB 15.7MB/s 00:00` —— 同一个数字。所以「以前传得快」是错觉，
一直是这个速度。

**根因**：KNC **没有 AES-NI**（`/proc/cpuinfo` 里连 `sse2` 都没有，它是独立 ISA），
SSH 加密全靠 1.2 GHz 单核软件实现。换 `chacha20-poly1305` 只能快 1.6x（13.6 vs 8.2 MB/s）——
**在最慢的通道上做微调，方向本身就错了**。

---

## 可用配方：`micnativeloadex`（走 SCIF，不经 SSH）

关键是要把**宿主侧**的 k1om sysroot 库路径喂给它做依赖解析：

```powershell
$ml  = 'C:\Program Files\Intel\MPSS\bin\micnativeloadex.exe'
$sys = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux'
$omp = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\compiler\lib\intel64_win_mic'

# 注意变量名是 SINK_LD_LIBRARY_PATH（不是 LD_LIBRARY_PATH）
$env:SINK_LD_LIBRARY_PATH = "$sys\lib64;$sys\usr\lib64;$omp"

& $ml D:\zigtest\k1om\bonsai_k1om -a "/tmp/bonsai.gguf selftest" -t 900
```

**验证结果**：依赖全部解析成功（含 `libiomp5.so`），远程进程创建成功、
程序在卡上执行、结果回传，**总耗时 1.4 秒**。

用 `-l` 可以先只列出依赖，不执行：

```powershell
& $ml D:\zigtest\k1om\bonsai_k1om -l
```

### 踩过的坑

1. **变量名是 `SINK_LD_LIBRARY_PATH`**，用 `LD_LIBRARY_PATH` 无效。
2. 它**在宿主侧解析依赖**，需要指向 k1om sysroot
   （`k1om-mpss-linux\lib64`），不是卡上的 `/lib64`。
3. 缺 `libiomp5.so` 时要额外加上 `intel64_win_mic` 目录 ——
   它会把这个 k1om 版的 `.so` 一起送过去。
4. `micnativeloadex` **只接受自己的路径做执行**，不能代理卡上读宿主文件。

---

## 仍未解决：怎么让 GB/s 级通道也覆盖模型文件

`micnativeloadex` 解决了**执行**（二进制 + 依赖），但 6.7 GB 的**模型**仍是问题：

- 它跑起来后，模型文件由**卡自己**从 `/tmp` 读 —— 所以模型还是得先到卡上
- 而 `/tmp` 是 tmpfs，落在卡的 GDDR5 上，**重启即清空**（本轮被清了两次）
- 目前唯一的填充方式是 SSH（9–15 MB/s，7.7 分钟）

### 可探索的方向（按前景排序）

1. **COI（Coprocessor Offload Infrastructure）** —— 这才是设计用来传数据的
   - 宿主侧头文件：`MPSS\k1om-mpss-linux\usr\include\intel-coi\source\COIBuffer_source.h`
   - 卡侧：`intel-coi\sink\`、`/usr/bin/coi_daemon`、`/usr/lib64/libcoi_device.so.0`
   - 文档：`MPSS\docs\coi\MIC_COI_API_Reference_Manual_1_0.pdf`
   - SCIF 已确认在跑：卡上内核线程 `SCIF INTR 0` / `SCIF_MISC` / `SCIF_NB_CONN`，
     设备节点 `/dev/mic`
   - 思路：用 `COIBuffer` 把宿主的模型文件映射成卡的可见缓冲，卡直接按需读
2. **`mic0.filelist`** —— 试过，`micctrl -g` 确实把文件烘焙进镜像
   （`mic0.image` 29.7 MB → 157.7 MB），但**重启后文件不出现**，
   说明 `-g` 重建的镜像不是卡实际启动用的那个
   （`initramfs-knightcorner.cpio.gz` 仍是 2019 年的 22.2 MB）。
   要查清卡到底从哪个镜像启动。
3. **`micctrl --addcifs`** —— 挂 CIFS 网络共享，但走的仍是虚拟以太网，不会更快。

---

## 一个可能顺带解决的更大问题

推理时每 token 要流式读 **6.8 GB 权重**。如果模型能通过 SCIF/COI
**从宿主内存按需映射**，那么：

- 传输时间 → 0（不再需要"上传"）
- 卡的 16 GB 内存腾出 6.8 GB
- 每次前向的权重读取也走 GB/s 级通道

这可能同时改善**部署**和**推理吞吐**。
