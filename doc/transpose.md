# Transpose

## 学习目标
- 写出正确的矩阵转置 CUDA kernel。
- 从 1D 索引过渡到 2D grid / block 映射。
- 观察 naive transpose 中“读连续、写跨 stride”的访存问题。
- 为下一轮 shared memory tile 和 bank conflict 分析建立基线。

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
下一版使用 **shared memory tiled transpose**。核心思路是先把一个 tile 连续读入 shared memory，再换方向连续写出到 global memory。

### v1 作业
1. 合上代码，自己默写 `kernel_naive` 的核心逻辑：计算 `row/col`、做二维边界检查、完成 `B[col * rows + row] = A[row * cols + col]`。
2. 用一段话解释：为什么 naive transpose 的读侧通常较连续，但写侧通常不连续？

## 对比总表
| version | ms | GB/s | TFLOPS | 说明 |
| --- | --- | --- | --- | --- |
| naive | 7.6666 | 17.51 | 0.0000 | 一个线程搬运一个元素，读连续、写跨 stride |

## 参考资料
- CUDA C++ Programming Guide: Device Memory Accesses
- NVIDIA CUDA Samples: matrixTranspose
- Nsight Compute: Memory Workload Analysis
