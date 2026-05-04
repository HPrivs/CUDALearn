# GEMM

## 学习目标
- 写出正确的 float32 GEMM CUDA kernel。
- 理解二维输出矩阵如何映射到 `grid/block/thread`。
- 建立 naive GEMM 的 correctness、benchmark 和定量分析基线。
- 学习 shared memory tiling：把输出 tile 内重复使用的 `A/B` 数据缓存到 shared memory。

## 前置知识
- GEMM：General Matrix Multiplication，通用矩阵乘法。
- dot product：两个长度为 `K` 的向量逐元素相乘再求和。
- row-major：二维数组按行连续存储，`A[row, col]` 的线性下标是 `row * cols + col`。
- output tile：一个 thread block 覆盖的 `C` 矩阵小矩形区域。
- shared memory：同一个 thread block 内线程共享的片上存储，延迟通常低于 global memory，但容量有限且需要显式搬运数据。
- `__syncthreads()`：thread block 内的同步屏障，保证所有线程都到达该点后才继续执行。

## 问题规格
- 输入：矩阵 `A`，形状为 `M x K`
- 输入：矩阵 `B`，形状为 `K x N`
- 输出：矩阵 `C`，形状为 `M x N`
- dtype：`float32`
- 数学定义：`C[row, col] = sum_k A[row, k] * B[k, col]`
- 默认规模：`M = 512, N = 512, K = 512`
- 存储布局：`A/B/C` 都是 row-major

v1 先实现最小正确版本：一个线程计算一个 `C[row, col]`。v2 引入 shared memory tile，但仍不做 register tile，也不调用 cuBLAS。

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

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。v2 本轮多次运行，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4663 | 240.64 | 0.0601 | 0.000034 |

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

## v2 — shared memory tile

### 本版学习目标
本轮唯一新增核心概念是 **shared memory tiling**：一个 block 负责一个 `16 x 16` 输出 tile，每次把 `K` 方向上一段 `A` tile 和 `B` tile 搬到 shared memory，让 block 内线程复用后再进入下一段。

### 改了什么
v2 保持「一个线程计算一个 `C[row, col]`」不变，只改变 `A/B` 的读取方式：

- `tile_a[16][16]` 缓存当前输出 tile 需要的 16 行 `A` 数据。
- `tile_b[16][16]` 缓存当前输出 tile 需要的 16 列 `B` 数据。
- `tile_k = 0, 16, 32, ...` 分阶段沿 `K` 方向推进。
- 每个阶段先把 global memory 的 `A/B` 片段搬进 shared memory，再用 `__syncthreads()` 保证所有线程都加载完成。
- 每个线程用 shared memory 中的 16 对元素做乘加，累加到自己的 `sum`。

### 为什么可能更快
naive 版中，同一个输出 tile 内的 `A[row, kk]` 会被同一行的 16 个输出列使用，`B[kk, col]` 会被同一列的 16 个输出行使用。v2 把这些会重复使用的数据先放进 shared memory，目标是把重复 global load 变成更便宜的 shared memory load。

这不是减少 FLOPs，而是减少有效 global memory 读。对默认 `16 x 16` 输出 tile，每个 `kk` 原来逻辑上要为 256 个输出读 `256` 个 `A` 和 `256` 个 `B`；v2 只需要读 `16` 个 `A` 和 `16` 个 `B`，然后在 tile 内复用。

### 代码要点
- 不能在 kernel 开头对越界线程提前 `return`，因为所有线程都必须参与后面的 `__syncthreads()`。
- 越界的 `A/B` load 写入 `0.0f` 到 shared memory，这样非整除的 `M/N/K` 也能得到正确结果。
- 每个 `tile_k` 阶段有两次同步：一次保证 shared memory 已写完，一次保证本阶段计算读完后再覆盖 shared memory。
- 最后写回 `C` 时才检查 `row < m && col < n`。

### 定量分析
v2 的 FLOPs 不变，仍是每个输出元素 `K` 次乘法和 `K` 次加法。变化在有效 global memory 读写量。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `(M*K*ceil(N/16) + K*N*ceil(M/16) + M*N) * 4` | `A` 对每个输出 tile 列读一次，`B` 对每个输出 tile 行读一次，`C` 写一次 |
| FLOPs `F` | `M * N * 2K` | 与 v1 相同 |
| 算术强度 `AI` | `F / B` | 默认规模约 `3.94 FLOP/Byte` |

默认规模下，v2 的有效访存 `B = 68,157,440 bytes`，约 `68.16 MB`；v1 naive 逻辑访存是约 `1.075 GB`。有效访存口径下，`A/B` global load 下降接近 `16x`，算术强度从约 `0.25 FLOP/Byte` 提高到约 `3.94 FLOP/Byte`。

瓶颈定性判定：**less memory-bound, but still far from compute-bound**。实测 `TFLOPS` 从 `0.0601` 提高到 `0.1325`，说明减少 global load 有效；但加速只有约 `2.21x`，没有接近逻辑访存下降比例。原因是 naive 版已有硬件 cache 自动复用一部分数据，v2 又引入 shared memory 读写、每个 `tile_k` 两次 `__syncthreads()`，并且仍然是每个线程只算一个输出元素。

可验证的 NCU metric 名继续沿用 v1 已列出的 global memory 和 SM 指标，例如 `dram__bytes_read`、`dram__bytes_write`、`dram__throughput`、`sm__throughput`、`smsp__inst_executed_op_global_ld` 和 `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`。本机执行 `ncu --query-metrics` 返回 `Skipping unsupported chip GP108`，因此本轮没有新增 shared-memory 专项 metric 名，也没有写入 profiler 实测数据。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。本轮多次运行，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4663 | 240.64 | 0.0601 | 0.000034 |
| v2_smem | 2.0254 | 33.65 | 0.1325 | 0.000034 |

这里 `GB/s` 使用各版本自己的有效访存字节计算。v2 的 `GB/s` 数字更低，不代表带宽利用更差；它的分母已经从约 `1.075 GB` 降到约 `68.16 MB`。跨版本更适合先看 `ms` 和 `TFLOPS`：v2 比 naive 快约 `2.21x`。

### 当前瓶颈
v2 主要瓶颈从「重复 global memory load」转向：

- 同步开销：`K=512, TILE_K=16` 时每个 block 有 32 个阶段，每阶段 2 次 `__syncthreads()`。
- shared memory 和 register 数据复用仍浅：每个线程只计算一个输出元素，没有 register tile。
- tile 参数固定：`16 x 16` 易懂，但未分析 occupancy、register pressure 和不同 tile shape 的取舍。

### 代价或限制
shared memory tile 增加了代码复杂度，并要求所有线程按相同路径经过同步点。它适合 `A/B` 在 block 内有明确复用的场景；如果 tile 太小、同步太频繁，或者硬件 cache 已经能很好复用，收益会明显低于理论访存下降比例。

### 下一步
下一轮建议做 register tile：让一个线程计算多个相邻输出元素，在 register 中累加，提高每次 shared memory load 的复用深度，同时观察 register pressure 对 occupancy 的影响。

### v2 作业
1. 概念题：默认 `TILE=16` 时，推导一个 `16 x 16` 输出 tile 在 naive 和 v2 中对 `A/B` 的 global load 数量。为什么 `C` 的写回次数没有减少？
2. 改错题：下面这段 shared memory tile 代码有什么问题？至少指出两个 bug，并说明会导致死锁、越界还是结果错误。

```cpp
if (row >= m || col >= n) return;
for (int tile_k = 0; tile_k < k; tile_k += 16) {
    tile_a[ty][tx] = a[row * k + tile_k + tx];
    tile_b[ty][tx] = b[(tile_k + ty) * n + col];
    __syncthreads();
    for (int inner = 0; inner < 16; inner++) {
        sum += tile_a[ty][inner] * tile_b[inner][tx];
    }
}
```

## 对比总表

| version | 核心手段 | 逻辑读写 | ms | GB/s | TFLOPS | max_err | 瓶颈 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程计算一个 `C[row, col]` | 每个输出读 `A/B` 各 `K` 次，写 `C` 1 次 | 4.4663 | 240.64 | 0.0601 | 0.000034 | memory-bound + no explicit data reuse |
| v2_smem | shared memory tile | `A/B` 按输出 tile 复用，默认有效访存约 `68.16 MB` | 2.0254 | 33.65 | 0.1325 | 0.000034 | less memory-bound + sync/shared-memory overhead |

## 参考资料
- CUDA C++ Programming Guide：thread hierarchy、global memory coalescing、shared memory。
- NVIDIA Nsight Compute CLI：后续 profiler 验证使用 `ncu` metric。
