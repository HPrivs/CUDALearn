# GEMM

## 学习目标
- 写出正确的 float32 GEMM CUDA kernel。
- 理解二维输出矩阵如何映射到 `grid/block/thread`。
- 建立 naive GEMM 的 correctness、benchmark 和定量分析基线。
- 为下一步 shared memory tile 做准备：先看清 naive 版哪里重复读取了 `A/B`。

## 前置知识
- GEMM：General Matrix Multiplication，通用矩阵乘法。
- dot product：两个长度为 `K` 的向量逐元素相乘再求和。
- row-major：二维数组按行连续存储，`A[row, col]` 的线性下标是 `row * cols + col`。
- output tile：一个 thread block 覆盖的 `C` 矩阵小矩形区域。

## 问题规格
- 输入：矩阵 `A`，形状为 `M x K`
- 输入：矩阵 `B`，形状为 `K x N`
- 输出：矩阵 `C`，形状为 `M x N`
- dtype：`float32`
- 数学定义：`C[row, col] = sum_k A[row, k] * B[k, col]`
- 默认规模：`M = 512, N = 512, K = 512`
- 存储布局：`A/B/C` 都是 row-major

本轮只实现最小正确版本：一个线程计算一个 `C[row, col]`。不使用 shared memory，不做 register tile，也不调用 cuBLAS。

## v1 — naive

### 本版学习目标
先把 GEMM 的二维索引和 dot product 写正确，建立后续优化的性能基线。

### 改了什么
首版使用二维 thread block：

- `blockDim = (16, 16)`，一个 block 负责 `C` 上一个 `16 x 16` 的输出区域。
- `col = blockIdx.x * blockDim.x + threadIdx.x`。
- `row = blockIdx.y * blockDim.y + threadIdx.y`。
- 每个有效线程串行循环 `kk = 0..K-1`，累加一个 dot product。
- 写回 `C[row, col]`。

### 为什么可能更快
这是 correctness 基线，不追求高性能。它的价值是把矩阵乘法的输出映射、边界检查、CPU reference 和 benchmark 跑通。

和 GEMV 不同，GEMM 的每个 `A[row, k]` 理论上可以被同一行的多个输出列复用，每个 `B[k, col]` 也可以被同一列的多个输出行复用。naive 版没有显式利用这种复用；下一步 shared memory tile 会专门解决这个问题。

### 代码要点
- `grid.x` 覆盖 `N` 方向，`grid.y` 覆盖 `M` 方向。
- 边界条件必须检查 `row >= m || col >= n`，不能假设矩阵尺寸一定能整除 block shape。
- `A` 的下标是 `row * k + kk`，`B` 的下标是 `kk * n + col`，`C` 的下标是 `row * n + col`。
- CPU reference 用 `double` 累加后转回 `float`，降低参考结果自身误差。

### 定量分析
按 naive 逻辑访存计算，每个输出元素会读取 `K` 个 `A` 元素、`K` 个 `B` 元素，并写 1 个 `C` 元素。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * N * (2K + 1) * 4` | 每个 `C` 读 `K` 个 `A`、`K` 个 `B`，写 1 个 `C` |
| FLOPs `F` | `M * N * 2K` | 每个 `kk` 统计 1 次乘法 + 1 次加法 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 这是 naive 逻辑访存口径，不是最优 GEMM 的数据复用口径 |

默认规模下，`B = 1,074,790,400 bytes`，约 `1.075 GB`；`F = 268,435,456 FLOPs`，约 `268.4 MFLOPs`。

瓶颈定性判定：**memory-bound + no explicit data reuse**。naive 版为每个输出元素重复从 global memory 读取 `A/B`，没有把一个 tile 内会重复使用的数据缓存到 shared memory。实际硬件 cache 可能自动复用一部分数据，所以 profiler 里的 DRAM bytes 未必等于上面的逻辑字节；本项目的 `GB/s` 仍按有效逻辑访存字节计算，便于版本间对比。

可验证的 NCU metric 名沿用项目前文已查询确认的指标：
- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`

本文档没有写入 profiler 实测数据。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。本轮重复运行 4 次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.6087 | 233.21 | 0.0582 | 0.000034 |

本次 `max_err = 3.43323e-05`，低于代码里的 `1e-4` correctness 阈值。误差来自 GPU float 累加和 CPU double reference 后转回 float 的差异，量级符合当前 `K = 512` 的预期。

`TFLOPS` 很低，说明 naive GEMM 远没有接近 GPU 的矩阵乘吞吐潜力。当前主要问题不是公式，而是数据复用：一个 `A/B` 元素会被多个相邻输出复用，但 naive 版让不同线程各自从 global memory 读取。

### 当前瓶颈
naive 版有三个明显限制：

- 重复读取：相邻输出会复用同一批 `A/B` 数据，但代码没有显式缓存。
- 单线程长循环：每个线程串行完成长度为 `K` 的 dot product。
- 写回简单但计算密度低：按 naive 逻辑字节计算，`AI` 只有约 `0.25 FLOP/Byte`。

### 代价或限制
代码简单，边界完整，适合作为正确性基线。代价是没有 shared memory tiling，也没有 register tiling；随着 `M/N/K` 增大，重复 global read 会成为主要浪费。

### 下一步
下一轮建议做 shared memory tile：一个 block 负责一个 `C` tile，把对应的 `A/B` tile 分阶段搬到 shared memory，让 tile 内多个线程复用这些数据。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row/col`、边界检查、循环 `kk` 做 dot product、写回 `C[row, col]`。
2. 用一段话解释：为什么 naive GEMM 没有利用好数据复用？请分别说明 `A[row, kk]` 和 `B[kk, col]` 在一个输出 tile 内本来可以怎样被复用。

## 对比总表

| version | 核心手段 | 逻辑读写 | ms | GB/s | TFLOPS | max_err | 瓶颈 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程计算一个 `C[row, col]` | 每个输出读 `A/B` 各 `K` 次，写 `C` 1 次 | 4.6087 | 233.21 | 0.0582 | 0.000034 | memory-bound + no explicit data reuse |

## 参考资料
- CUDA C++ Programming Guide：thread hierarchy、global memory coalescing、shared memory。
- NVIDIA Nsight Compute CLI：后续 profiler 验证使用 `ncu` metric。
