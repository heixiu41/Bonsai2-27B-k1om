# COI 路线：现状、阻塞点与下一步

## 目标

把宿主机上的 6.7 GB 模型通过 **COI（Coprocessor Offload Infrastructure）** 映射给卡按需读取，
替代目前走 SSH 虚拟以太网的 9–15 MB/s 慢通道。

若成功，一次解决三件事：
1. **传输时间 → 0**（不再需要"上传"模型）
2. **卡腾出 6.8 GB 内存**（模型留在宿主）
3. **推理时的权重读取也走 GB/s 级通道**（每 token 要流式读 6.8 GB）

## 已确认的可用资源

### 宿主侧
```
C:\Windows\System32\coi_host.dll                             420 KB   运行时
C:\Program Files\Intel\MPSS\sdk\lib\coi_host.lib             导入库
C:\Program Files\Intel\MPSS\sdk\include\intel-coi\source\*.h  头文件
  COIEngine_source.h  COIEvent_source.h  COIBuffer_source.h
  COIPipeline_source.h  COIProcess_source.h
C:\Program Files\Intel\MPSS\sdk\include\scif.h                底层 SCIF
```

### 卡侧
```
k1om-mpss-linux\usr\lib64\libcoi_device.so                    COI 设备库
k1om-mpss-linux\usr\include\intel-coi\sink\*.h                sink 侧头文件
/usr/bin/coi_daemon                                           运行中
/usr/lib64/libcoi_device.so.0
/dev/mic                                                      SCIF 设备节点
```
SCIF 已确认在跑：内核线程 `SCIF INTR 0` / `SCIF_MISC` / `SCIF_NB_CONN`。

### 可编译样例（模板，不必从零写）
```
MPSS\sdk\tutorials\coi\
  README_Windows_1.txt
  win_build.bat
  buffer_with_user_memory\        ★ 最对口：宿主内存映射
  hello_world\  coi_simple\  buffer_references\
  buffers_with_pipeline_function\
  multiple_pipeline_explicit\  multiple_pipeline_implicit\  user_event\
```

### 文档
```
MPSS\docs\coi\MIC_COI_API_Reference_Manual_1_0.pdf   142 页
```

## 已确认的关键 API

```c
// 用调用者的内存建一个 COI 缓冲（不做拷贝）
COIResult COIBufferCreateFromMemory(
    uint64_t in_Size, COI_BUFFER_TYPE in_Type, uint32_t in_Flags,
    void *in_Memory, uint32_t in_NumProcesses, const COIPROCESS *in_pProcesses,
    COIBUFFER *out_pBuffer);

COIBufferMap(COIBUFFER, uint64_t offset, uint64_t length, COI_MAP_TYPE,
             uint32_t nDeps, const COIEVENT *deps, COIMAPINSTANCE *out_pMap,
             void **out_ppData, COIEVENT *out_pEvent);

COIBufferGetSinkAddress(COIBUFFER in_Buffer, uint64_t *out_pAddress);
COIBufferRead / COIBufferWrite / COIBufferCopy
COIProcessCreateFromFile / COIProcessLoadLibraryFromFile
COIPipelineStartExecutingRunFunctions / COIPipelineRunFunction
COIEngineGetCount / COIEngineGetHandle
```

`COI_SINK_MEMORY` 标志用于 sink 侧内存。

## ✅ 阻塞已解除：用 Python ctypes 直接驱动 COI（不需要 MSVC）

**关键发现**：`coi_host.dll` 导出 **71 个函数，全是标准 C ABI 名字**（无 C++ 修饰），
所以宿主侧 COI 可以用 Python `ctypes` 完全驱动 —— **不需要编译器，不需要 Visual Studio**。

### 已实测成功（`scripts/coi_probe.py`）

```
COIEngineGetCount(COI_DEVICE_MIC) rc=0x00000000  count=1
COIEngineGetCount(COI_DEVICE_KNC) rc=0x00000000  count=1
COIEngineGetHandle(MIC, 0)        rc=0x00000000  handle=140721241686448
```

### 类型定义必须照 SDK 头文件来（我一开始猜错了两个）

`COITypes_common.h:68-75` —— 全是**不透明指针**，不是结构体：
```c
typedef struct coiengine   *COIENGINE;    // c_void_p
typedef struct coiprocess  *COIPROCESS;
typedef struct coipipeline *COIPIPELINE;
typedef struct coibuffer   *COIBUFFER;
typedef struct coimapinst  *COIMAPINSTANCE;
struct coievent { uint64_t opaque[2]; };  // 这个是值类型
```

`COIEngine_common.h:37-48` —— 设备类型枚举：
```c
COI_DEVICE_INVALID = 0,
COI_DEVICE_SOURCE,          // 1  ← 宿主，传它得到 INVALID_PARAMETER
COI_DEVICE_MIC,             // 2  ← 用这个
COI_DEVICE_DEPRECATED_0,    // 3
COI_DEVICE_KNC,             // 4  Knights Corner
COI_DEVICE_KNL,             // 5
```

### 所以 COI 实际需要的东西

| 侧 | 需要 | 状态 |
|---|---|---|
| 卡 | `icc -mmic` + sysroot + `sink\*.h` + `libcoi_device.so` + `coi_daemon` + SCIF | ✅ 全部就绪 |
| 宿主 | `coi_host.dll` + `source\*.h` + **Python ctypes** | ✅ 已打通 |
| ~~宿主~~ | ~~MSVC / Visual Studio~~ | ❌ **不需要了** |

## 下一步（不再依赖任何安装）

```
1. 扩展 scripts/coi_probe.py：
     COIProcessCreateFromFile   -- 在卡上起进程
     COIBufferCreateFromMemory  -- 用宿主内存建缓冲（宿主 mmap 模型文件）
     COIBufferGetSinkAddress    -- 拿卡侧虚拟地址
     COIPipelineCreate/RunFunction -- 在卡上跑函数
2. 先跑通"映射一小块宿主内存给卡读"的验证
3. 实测带宽（预期 1–2 GB/s，对照 SSH 的 13 MB/s）
4. 套到 Bonsai：宿主 mmap 6.71 GB → COI 缓冲 → 卡上 forward 按地址读
```

---

## 附：原始阻塞记录（已过时，保留作为参考）

COI 样例用 **`icl`** 构建，而 `icl` 依赖 MSVC 的 `link.exe`：

| 组件 | 状态 |
|---|---|
| `icl.exe` | ✅ `IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64\icl.exe` |
| `link.exe` / `cl.exe` / `lib.exe` / `nmake.exe` | ❌ **全都没有** |
| Visual Studio 14.0 | ❌ 未安装 |

`win_build.bat` 的实际命令：
```bat
icl /Qmic %%I_sink.cpp -c -o %%I_sink.o
icl /Qmic -rdynamic %%I_sink.o -o %%I_sink_mic -pthread -lcoi_device
copy ..\x64\Release\%%I_source.exe
```
`README_Windows_1.txt` 明确要求 **Visual Studio 2012**（x64）+ Intel Parallel Studio XE 2013。
版本不是硬性的，但**必须有 MSVC 的 link.exe**。

## 解法就在手边

```
D:\Users\Administrator\Desktop\windows\Intel_Parallel_Studio_XE_2017_win\vs2015.3.com_chs.iso
```
7.11 GB，**Visual Studio 2015 Update 3 中文版** —— PSXE 2017 自带它就是因为
Windows 侧编译需要 MSVC 后端。

## 下一步

**ISO 已挂载到 `E:`**（`Mount-DiskImage` 成功）：

```
E:\
  vs_community.exe      ← VS2015 Community 安装器
  packages\
  Standalone Profiler\
  autorun.inf
```

重挂命令（重启后需重新执行）：
```powershell
Mount-DiskImage -ImagePath 'D:\Users\Administrator\Desktop\windows\Intel_Parallel_Studio_XE_2017_win\vs2015.3.com_chs.iso'
```

```
2. 安装 VS2015 的 C++ 工具链 + Windows SDK（至少，不必装 IDE）
   静默安装（未验证，可能需要调整组件名）：
     E:\vs_community.exe /quiet /norestart ^
         /InstallSelectableItems NativeLanguageSupport_VC ^
         /ProductKey <key>
   或直接跑 GUI：E:\vs_community.exe
3. 编译并跑通 buffer_with_user_memory 样例：
     icl /Qmic buffer_with_user_memory_sink.cpp -c -o sink.o
     icl /Qmic -rdynamic sink.o -o sink_mic -pthread -lcoi_device
     icl buffer_with_user_memory_source.cpp -o source.exe ^
         /I"C:\Program Files\Intel\MPSS\sdk\include" /link coi_host.lib
4. 实测宿主内存 → 卡的带宽（预期 1–2 GB/s，对照 SSH 的 9 MB/s）
5. 套到 Bonsai：宿主 mmap 6.71 GB 模型 → COIBufferCreateFromMemory
   → COIBufferGetSinkAddress → 卡上 forward 直接按地址读
```

## 备选（不依赖 MSVC）

- **MinGW-w64 + dlltool**：从 `coi_host.dll` 生成 GNU 格式导入库，用 gcc 编译宿主侧。
  但 `coi_host.lib` 是 MSVC 格式，需要 `dlltool -k -d coi_host.def -l libcoi_host.a`，
  且 C++ 样例的 name mangling 会麻烦；纯 C 程序可行。
- **SCIF 直连**：`MPSS\sdk\include\scif.h` + `lib_uSCIF.lib`，比 COI 更底层，
  绕开 C++ 依赖，但需要自己管理注册/映射/DMA 描述符。
- **继续用 SSH**：加 `-c chacha20-poly1305@openssh.com`，13.6 MB/s，模型约 8 分钟。
  权宜之计，但至少可用。
