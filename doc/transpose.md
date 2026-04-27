# Transpose

## 学习目标
- 写出正确的矩阵转置 CUDA kernel。
- 从 1D 索引过渡到 2D grid / block 映射。
- 观察 naive transpose 中“读连续、写跨 stride”的访存问题。
- 用 shared memory tile 把 global memory 写入也组织成连续访问。
- 为下一轮 bank conflict 分析建立基线。

## 前置知识
- row-major：二维数组按行连续存储，`A[row, col]` 的线性下标是 `row * cols + col`。
- coalesced access：同一个 warp 内线程访问连续 global memory 地址时，硬件能把访问合并成更少的内存事务。
- strided access：相邻线程访问的地址间隔较大，通常会产生更多内存事务。

## 问题规格
- 输入：矩阵 `A`，形状为 `M x N`
- 输出：矩阵 `B`，形状为 `N x M`
- dtype：`float32`
- 数学定义：`B[col, row] = A[row, col]`
- 默认规模：`M = 4096, N = 4096`
- 存储布局：`A` 和 `B` 都是 row-major

Transpose 本身没有浮点计算。这个算子的价值在于观察同样是搬运 `float`，不同的二维访存方向会怎样影响有效带宽。

## v1 — naive
### 本版学习目标
建立 correctness 和 benchmark 基线：一个线程负责搬运一个元素。

### 改了什么
首版只做最直接的二维映射。`threadIdx.x` 对应列方向，`threadIdx.y` 对应行方向；每个线程读取一个 `A[row, col]`，写到 `B[col, row]`。

### 为什么可能更快
这是首版基线，不追求性能。它的作用是把二维索引、边界检查、CPU reference 和 benchmark 跑通。

### 代码要点
- `col = blockIdx.x * blockDim.x + threadIdx.x`
- `row = blockIdx.y * blockDim.y + threadIdx.y`
- 输入下标：`row * cols + col`
- 输出下标：`col * rows + row`
- 边界检查必须同时检查 `row < rows && col < cols`，不能假设矩阵尺寸一定整除 tile 大小

### 定量分析
对每个元素：

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * N * (4 + 4)` | 读 `A` 一个 `float`，写 `B` 一个 `float` |
| FLOPs `F` | `0` | 转置只搬运数据，不做浮点运算 |
| 算术强度 `AI` | `0 / B = 0 FLOP/Byte` | 没有浮点计算 |

瓶颈定性判定：**memory-bound / memory-transaction-bound**。理论上每个元素只需要 8B 有效访存，但 naive 写入 `B[col * rows + row]` 时，相邻线程的输出地址通常间隔 `rows * sizeof(float)`，写入不能很好 coalescing。

这个版本的读 `A[row * cols + col]` 对列方向是连续的，所以读侧通常较好；问题主要在写侧。下一轮会用 shared memory tile 把全局内存读写都组织成连续访问。

可验证的 NCU metric 名：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`

当前机器上 `ncu --query-metrics` 返回 `Skipping unsupported chip GP108`，所以本轮没有写入 profiler 实测结论。上面这些 metric 用于后续在 Nsight Compute 支持的 GPU 上验证：重点看 load/store sector 数量和 DRAM 吞吐。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/transpose.cu -o debugger/transpose && ./debugger/transpose
```

当前记录：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 7.6666 | 17.51 | 0.0000 | 0 |

如果实测有效 `GB/s` 明显低于 elementwise，不要先归因于“总字节更多”。更关键的是 naive transpose 的写入方向让同一个 warp 的 store 地址不连续，硬件需要更多内存事务来完成同样的有效字节。

### 当前瓶颈
主要瓶颈是 global memory 写入不连续。默认映射下，warp 内线程读 `A` 时大体连续，但写 `B` 时跨行跳跃，导致 store coalescing 差。

### 代价或限制
naive 版没有额外 shared memory，也没有同步，代码简单；代价是输出侧访存模式差，矩阵越大、stride 越大，这个问题越明显。

### 下一步
下一版继续分析 v2 的 shared memory 访问模式，重点看 tile 转置读取时可能出现的 bank conflict。

### v1 作业
1. 合上代码，自己默写 `kernel_naive` 的核心逻辑：计算 `row/col`、做二维边界检查、完成 `B[col * rows + row] = A[row * cols + col]`。
2. 用一段话解释：为什么 naive transpose 的读侧通常较连续，但写侧通常不连续？

## v2 — shared memory tiled transpose
### 本版学习目标
用 shared memory tile 把 naive 版的跨 stride global store 改成连续 global store。

本轮只新增一个核心概念：**shared memory tile**。它的作用是先在 block 内暂存一个二维小块，让线程可以用一种顺序连续读 global memory，再换一种顺序连续写 global memory。

### 改了什么
`v1` 每个线程直接执行：

```cpp
B[col * rows + row] = A[row * cols + col];
```

这样读 `A` 连续，但写 `B` 跨 stride。

`v2` 分成两步：
1. 先把输入 tile 按 `smem[threadIdx.y][threadIdx.x]` 读入 shared memory。
2. `__syncthreads()` 后，交换 `blockIdx.x / blockIdx.y` 对应的输出 tile，再从 `smem[threadIdx.x][threadIdx.y]` 读出并连续写入 `B`。

### 为什么可能更快
关键不是“用了 shared memory 就更快”，而是 global memory 的访问形状变了。

读取阶段，warp 内相邻线程访问：

```cpp
A[in_row * cols + in_col]
```

其中 `in_col` 连续变化，所以 load coalescing 较好。

写回阶段，`out_col = blockIdx.y * kTileDim + threadIdx.x`，相邻线程的 `threadIdx.x` 连续变化，所以访问：

```cpp
B[out_row * rows + out_col]
```

也是连续地址。shared memory 在中间承担“换方向”的作用。

### 代码要点
- load 坐标使用输入矩阵形状：`in_row < rows && in_col < cols`。
- store 坐标使用输出矩阵形状：`out_row < cols && out_col < rows`。
- `out_row/out_col` 里必须交换 block 坐标：输出 tile 的行来自输入 tile 的列块，输出 tile 的列来自输入 tile 的行块。
- `__syncthreads()` 不能省略，因为写回前必须保证整个 tile 已经读入 shared memory。

坐标关系可以按这张表记：

| 阶段 | 行坐标 | 列坐标 | 线性下标 |
| --- | --- | --- | --- |
| load `A` | `in_row = blockIdx.y * T + ty` | `in_col = blockIdx.x * T + tx` | `in_row * cols + in_col` |
| store `B` | `out_row = blockIdx.x * T + ty` | `out_col = blockIdx.y * T + tx` | `out_row * rows + out_col` |

这里 `T = kTileDim`，`tx = threadIdx.x`，`ty = threadIdx.y`。

### 定量分析
对每个元素的有效工作量没有变化：

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * N * (4 + 4)` | 读 `A` 一个 `float`，写 `B` 一个 `float` |
| FLOPs `F` | `0` | 转置只搬运数据，不做浮点运算 |
| 算术强度 `AI` | `0 / B = 0 FLOP/Byte` | 没有浮点计算 |

瓶颈定性判定：**memory-bound / memory-transaction-bound**。`v2` 没有减少有效字节数，仍然是每个元素 8B；它减少的是 naive 写侧跨 stride 造成的额外 memory transaction。

本版可以继续用这些 NCU metric 验证 global memory 侧变化：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`

当前机器上 `ncu --query-metrics` 返回 `Skipping unsupported chip GP108`，所以本轮仍没有写入 profiler 实测结论。用 Nsight Compute 支持的 GPU 验证时，重点比较 `v1` 和 `v2` 的 store sector 数量是否下降。

### 实测结果
编译运行命令：

```bash
nvcc src/transpose.cu -o debugger/transpose && ./debugger/transpose
```

当前记录，默认 `M = 4096, N = 4096`，`timeit` 使用 `warmup=3, iters=20`：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 7.6057 | 17.65 | 0.0000 | 0 |
| v2 | 4.2992 | 31.22 | 0.0000 | 0 |

相对当前机器的 `naive`，`v2` 约快 `1.77x`。这个提升来自 global store 访问更连续，而不是 FLOPs 变化；两个版本的 FLOPs 都是 0。

### 当前瓶颈
`v2` 主要瓶颈仍是 memory-bound。global memory 访问比 naive 更规整，但每个元素仍要读写一次 global memory，并且多了 shared memory 读写和一次 `__syncthreads()`。

另一个待验证问题是 shared memory bank conflict。当前 `smem[threadIdx.x][threadIdx.y]` 的读法会让同一个 warp 的线程按列方向读取 shared memory，下一版会专门处理这个问题。

### 代价或限制
- 多了一次 shared memory 写、一次 shared memory 读和一次 block 内同步。
- 当前版本没有 padding，可能存在 shared memory bank conflict。
- tile 大小固定为 `16 x 16`，还没有比较不同 tile size 对性能的影响。

### 下一步
下一版使用 **shared memory padding**，例如把 tile 声明成 `smem[kTileDim][kTileDim + 1]`，减少转置读取 shared memory 时的 bank conflict。

### v2 作业
1. 用自己的话解释：为什么 `v2` 写回 `B` 时要交换 `blockIdx.x` 和 `blockIdx.y`？如果不交换，会发生什么？
2. 改错题：下面代码有两个和 `v2` 相关的问题，请指出并说明后果。

```cpp
int out_col = blockIdx.x * kTileDim + threadIdx.x;
int out_row = blockIdx.y * kTileDim + threadIdx.y;

if (out_row < rows && out_col < cols) {
    b[out_row * cols + out_col] = smem[threadIdx.x][threadIdx.y];
}
```

## 对比总表
| version | ms | GB/s | TFLOPS | 说明 |
| --- | --- | --- | --- | --- |
| naive | 7.6057 | 17.65 | 0.0000 | 一个线程搬运一个元素，读连续、写跨 stride |
| v2 | 4.2992 | 31.22 | 0.0000 | shared memory tile，读写 global memory 都尽量连续 |

## 参考资料
- CUDA C++ Programming Guide: Device Memory Accesses
- NVIDIA CUDA Samples: matrixTranspose
- Nsight Compute: Memory Workload Analysis
