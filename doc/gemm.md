# GEMM

## 学习目标
- 写出正确的 float32 GEMM CUDA kernel。
- 理解二维输出矩阵如何映射到 `grid/block/thread`。
- 建立 naive GEMM 的 correctness、benchmark 和定量分析基线。
- 学习 shared memory tiling：把输出 tile 内重复使用的 `A/B` 数据缓存到 shared memory。
- 学习 register tile：一个线程维护多个寄存器累加器，提高 shared memory 数据的线程内复用。
- 学习 tile 参数与资源取舍：扩大 per-thread 输出 tile 时，观察有效访存、register pressure、shared memory 和 occupancy 的变化。
- 加入 cuBLAS SGEMM 对比基线，理解 row-major 输入如何映射到 cuBLAS 的 column-major 调用。

## 前置知识
- GEMM：General Matrix Multiplication，通用矩阵乘法。
- dot product：两个长度为 `K` 的向量逐元素相乘再求和。
- row-major：二维数组按行连续存储，`A[row, col]` 的线性下标是 `row * cols + col`。
- output tile：一个 thread block 覆盖的 `C` 矩阵小矩形区域。
- shared memory：同一个 thread block 内线程共享的片上存储，延迟通常低于 global memory，但容量有限且需要显式搬运数据。
- `__syncthreads()`：thread block 内的同步屏障，保证所有线程都到达该点后才继续执行。
- register tile：一个线程计算多个输出元素，并在 register 中保存多个 partial sum。
- register pressure：单个线程需要的寄存器数量；寄存器越多，可能让同一个 SM 同时驻留的 block/warp 变少。
- occupancy：SM 上同时驻留的 active warps 相对硬件上限的比例；它不是越高越快，但过低可能暴露延迟。

## 问题规格
- 输入：矩阵 `A`，形状为 `M x K`
- 输入：矩阵 `B`，形状为 `K x N`
- 输出：矩阵 `C`，形状为 `M x N`
- dtype：`float32`
- 数学定义：`C[row, col] = sum_k A[row, k] * B[k, col]`
- 默认规模：`M = 512, N = 512, K = 512`
- 存储布局：`A/B/C` 都是 row-major

v1 先实现最小正确版本：一个线程计算一个 `C[row, col]`。v2 引入 shared memory tile。v3 在 v2 基础上做 `2 x 1` register tile。v4 扩展到 `2 x 2` register tile。v5 是当前本机可验证的 FP32 标量收官版：`4 x 4` register tile。`cublas_sgemm` 是额外的库函数对比基线，不属于手写 kernel 优化版本。Tensor Core / `cp.async` 路径保留为后续参考，不放在当前简洁版 `gemm.cu` 中。

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
nvcc src/gemm.cu -o debugger/gemm -lcublas && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。v5 本轮重新运行 4 次，下表取 naive 的最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4468 | 241.70 | 0.0604 | 0.000034 |

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

瓶颈定性判定：**less memory-bound, but still far from compute-bound**。本轮重新实测时，`TFLOPS` 从 naive 的 `0.0604` 提高到 `0.1315`，说明减少 global load 有效；但加速只有约 `2.18x`，没有接近逻辑访存下降比例。原因是 naive 版已有硬件 cache 自动复用一部分数据，v2 又引入 shared memory 读写、每个 `tile_k` 两次 `__syncthreads()`，并且仍然是每个线程只算一个输出元素。

可验证的 NCU metric 名继续沿用 v1 已列出的 global memory 和 SM 指标，例如 `dram__bytes_read`、`dram__bytes_write`、`dram__throughput`、`sm__throughput`、`smsp__inst_executed_op_global_ld` 和 `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`。本机执行 `ncu --query-metrics` 返回 `Skipping unsupported chip GP108`，因此本轮没有新增 shared-memory 专项 metric 名，也没有写入 profiler 实测数据。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm -lcublas && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。v5 本轮重新运行 4 次，下表取每版最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4468 | 241.70 | 0.0604 | 0.000034 |
| v2_smem | 2.0413 | 33.39 | 0.1315 | 0.000034 |

这里 `GB/s` 使用各版本自己的有效访存字节计算。v2 的 `GB/s` 数字更低，不代表带宽利用更差；它的分母已经从约 `1.075 GB` 降到约 `68.16 MB`。跨版本更适合先看 `ms` 和 `TFLOPS`：v2 比 naive 快约 `2.18x`。

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

## v3 — 2x1 register tile

### 本版学习目标
本轮唯一新增核心概念是 **register tile**：一个线程不再只维护一个 `sum`，而是在 register 里维护多个 partial sum，让一次 shared memory 读取服务多个输出元素。

v3 采用最小改动的 `2 x 1` 形式：一个线程计算同一列上的两个 `C` 元素。这样可以重点观察寄存器累加带来的复用变化，而不同时引入 vectorized load、double buffering 或 Tensor Core。

### 改了什么
v2 的一个 block 负责 `16 x 16` 输出 tile，共 256 个输出元素。v3 仍使用 `blockDim = (16, 16)`，但一个 block 负责 `32 x 16` 输出 tile：

- `row0 = blockIdx.y * 32 + threadIdx.y`。
- `row1 = row0 + 16`。
- `col = blockIdx.x * 16 + threadIdx.x`。
- 每个线程维护 `sum0` 和 `sum1` 两个寄存器累加器。
- shared memory 变为 `tile_a[32][16]` 和 `tile_b[16][16]`。

### 为什么可能更快
v2 中每个线程只算一个输出元素。进入 `inner` 循环时，线程从 shared memory 读出一个 `B[inner, col]` 后只用于一次乘加。

v3 中同一个线程计算两个同列输出：

```cpp
const float b_val = tile_b[inner][tx];
sum0 += tile_a[ty][inner] * b_val;
sum1 += tile_a[ty + 16][inner] * b_val;
```

这让一次 `tile_b` shared load 服务两个累加器。输出 tile 的高度也从 16 变成 32，所以同一块 `B` tile 被 32 行输出复用，默认规模下 `B` 的有效 global load 次数减半。

### 代码要点
- v3 不能让 `row1` 越界线程提前 `return`，因为整个 block 后面仍然要经过 `__syncthreads()`。
- `row0` 和 `row1` 分别做边界检查；越界的 `A` 元素写 `0.0f` 到 shared memory。
- `B` tile 仍是 `16 x 16`，由所有线程按 `tile_b[ty][tx]` 搬运。
- 本版只做 `2 x 1` register tile，复用的是 `B` 值；如果改成 `1 x 2`，复用方向会变成 `A` 值。

### 定量分析
v3 的 FLOPs 仍然不变，每个输出元素还是 `K` 次乘法和 `K` 次加法。变化在有效 global memory 读写量和 shared memory 读取复用。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `(M*K*ceil(N/16) + K*N*ceil(M/32) + M*N) * 4` | `A` 对每个输出 tile 列读一次，`B` 对每个 32 行输出 tile 读一次，`C` 写一次 |
| FLOPs `F` | `M * N * 2K` | 与 v1/v2 相同 |
| 算术强度 `AI` | `F / B` | 默认规模约 `5.22 FLOP/Byte` |

默认规模下，v3 的有效访存 `B = 51,380,224 bytes`，约 `51.38 MB`；v2 是约 `68.16 MB`。v3 相比 v2 主要减少的是 `B` 的有效 global load：`ceil(M/16)` 个输出行 tile 变成 `ceil(M/32)` 个输出行 tile。

瓶颈定性判定：**less memory-bound + register/shared-memory reuse, still sync-limited**。本轮实测 `TFLOPS` 从 v2 的 `0.1315` 提高到 `0.2142`，说明 register tile 提高复用有效；但它仍然远低于理想 GEMM。主要原因是每个 `K` tile 仍有两次 `__syncthreads()`，每个线程只做两个输出，tile 参数也还没有围绕 occupancy 和 register pressure 调整。

可验证的 NCU metric 名：

- `smsp__inst_executed_op_shared_ld`
- `smsp__inst_executed_op_shared_st`
- `smsp__warp_issue_stalled_barrier_per_warp_active`
- `dram__bytes_read`
- `dram__bytes_write`
- `sm__throughput`

其中前三个已用 `ncu --chips ga100 --query-metrics` 查询确认。本机实际 GPU 仍无法采集 Nsight Compute profiler 数据。另用 `nvcc -arch=sm_61 --ptxas-options=-v src/gemm.cu -o debugger/gemm_ptxas_sm61 -lcublas` 查看编译资源：`kernel_v2` 使用 27 registers、2048B shared memory；`kernel_v3` 使用 31 registers、3072B shared memory，且没有 spill。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm -lcublas && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。v5 本轮重复运行 4 次，下表取每版最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4468 | 241.70 | 0.0604 | 0.000034 |
| v2_smem | 2.0413 | 33.39 | 0.1315 | 0.000034 |
| v3_reg_tile | 1.2530 | 41.01 | 0.2142 | 0.000034 |

相对本轮重新测得的 v2，v3 约快 `2.0413 / 1.2530 = 1.63x`。相对 naive，v3 约快 `3.55x`。

### 当前瓶颈
v3 已经比 v2 更快，但仍有几个明显限制：

- 每个 `tile_k` 阶段仍有两次 block-level 同步。
- `2 x 1` register tile 只复用 `B`，没有同时做 `M x N` 的二维 register tile。
- register 数从 v2 的 27 增加到 31；当前没有 spill，但更大的 register tile 可能降低 occupancy。
- shared memory 仍按标量方式加载，没有 vectorized load，也没有 double buffering。

### 代价或限制
register tile 用更多寄存器和更多 shared memory 换更高的数据复用。`2 x 1` 版本比较适合教学，因为索引变化小；但它不是最终 GEMM 形态。继续增大每线程输出数量时，需要同时关注 register pressure、occupancy、shared memory 容量和访存指令形状。

### 下一步
下一节 v4 进入 tile 参数与 occupancy/register pressure 分析：用 `2 x 2` register tile 作为最小扩展，观察 register 数、shared memory、block 数和实际性能之间的取舍。

### v3 作业
1. 推导题：v3 的输出 tile 是 `32 x 16`。请推导一个输出 tile 对 `A/B` 的 global load 数量，并说明为什么相对 v2 主要减少的是 `B` 读取而不是 `A` 读取。
2. 预测题：如果把 v3 改成 `1 x 2` register tile，一个线程计算同一行的两个相邻列，理论上会减少 `A` 还是 `B` 的有效 global load？它可能引入哪些新的代价？

## v4 — 2x2 register tile

### 本版学习目标
本轮唯一新增核心概念是 **tile 参数与 register pressure 取舍**：把 per-thread 输出从 `2 x 1` 扩到 `2 x 2`，观察有效 DRAM 访存继续下降时，寄存器和 shared memory 资源如何上升。

v4 不是最终 GEMM 优化形态。它的价值是把“更大 tile 通常更能复用数据”这句话落到可检查的事实：少读了多少 global memory、多用了多少 register/shared memory、实际有没有变快。

### 改了什么
v4 仍使用 `blockDim = (16, 16)`，但一个 block 负责 `32 x 32` 输出 tile：

- `row0 = blockIdx.y * 32 + threadIdx.y`。
- `row1 = row0 + 16`。
- `col0 = blockIdx.x * 32 + threadIdx.x`。
- `col1 = col0 + 16`。
- 每个线程维护 `sum00/sum01/sum10/sum11` 四个寄存器累加器。
- shared memory 变为 `tile_a[32][16]` 和 `tile_b[16][32]`。

每个 `tile_k` 阶段中，每个线程加载两行 `A` 和两列 `B`：

```cpp
tile_a[ty][tx] = A[row0, tile_k + tx];
tile_a[ty + 16][tx] = A[row1, tile_k + tx];
tile_b[ty][tx] = B[tile_k + ty, col0];
tile_b[ty][tx + 16] = B[tile_k + ty, col1];
```

### 为什么可能更快
v3 的输出 tile 是 `32 x 16`，所以 `B` 被 32 行复用，但 `A` 仍只被 16 列复用。v4 把输出 tile 扩到 `32 x 32`，让 `A` 和 `B` 两侧都按 32 个输出复用。

在 `inner` 循环中，一个线程读出两个 `A` 值和两个 `B` 值，形成 4 次乘加：

```cpp
sum00 += a0 * b0;
sum01 += a0 * b1;
sum10 += a1 * b0;
sum11 += a1 * b1;
```

这比 v3 更充分地复用 shared memory load，但代价是更多 accumulator、更多 shared memory、更多每线程指令。

### 代码要点
- 不能只让 256 个线程各加载一个 `B`，因为 `tile_b` 是 `16 x 32`，共有 512 个元素；本版让每个线程加载 `tx` 和 `tx + 16` 两列。
- `tile_a` 也是 512 个元素，每个线程加载 `row0` 和 `row1` 两个位置。
- 四个输出各自做边界检查；越界 load 写 `0.0f` 到 shared memory，避免非整除规模出错。
- `__syncthreads()` 的位置和 v2/v3 一样：加载完 shared memory 后同步，当前 tile 使用完后再同步，防止下一轮覆盖。

### 定量分析
v4 的 FLOPs 仍然不变，每个输出元素还是 `K` 次乘法和 `K` 次加法。变化在输出 tile 从 `32 x 16` 变成 `32 x 32` 后，`A` 的有效 global load 也减半。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `(M*K*ceil(N/32) + K*N*ceil(M/32) + M*N) * 4` | `A` 对每个 32 列输出 tile 读一次，`B` 对每个 32 行输出 tile 读一次，`C` 写一次 |
| FLOPs `F` | `M * N * 2K` | 与 v1/v2/v3 相同 |
| 算术强度 `AI` | `F / B` | 默认规模约 `7.76 FLOP/Byte` |

默认规模下，v4 的有效访存 `B = 34,603,008 bytes`，约 `34.60 MB`。拆开看：

| 项 | v3 | v4 | 变化 |
| --- | ---: | ---: | --- |
| `A` 有效读取 | 32 MiB | 16 MiB | 输出 tile 宽度从 16 变 32，减半 |
| `B` 有效读取 | 16 MiB | 16 MiB | 输出 tile 高度仍是 32，不变 |
| `C` 写回 | 1 MiB | 1 MiB | 输出元素数不变 |

瓶颈定性判定：**更高 AI，但 register/shared-memory pressure 更高**。v4 相比 v3 的有效访存从约 `51.38 MB` 降到 `34.60 MB`，理论上减少约 `1.48x`；本轮实测时间从 `1.2530 ms` 降到 `0.8855 ms`，约 `1.42x`。收益接近但小于有效访存下降比例，说明更大 register tile 有效，但新增寄存器、shared memory 读写和指令也开始构成代价。

用 `nvcc -arch=sm_61 --ptxas-options=-v src/gemm.cu -o debugger/gemm_ptxas_sm61 -lcublas` 查看编译资源：

| kernel | registers/thread | shared memory/block | spill |
| --- | ---: | ---: | ---: |
| `kernel_v2` | 27 | 2048B | 0B |
| `kernel_v3` | 31 | 3072B | 0B |
| `kernel_v4` | 40 | 4096B | 0B |
| `kernel_v5` | 64 | 8192B | 0B |

资源判断要分清事实和估算：ptxas 明确给出 v4/v5 都没有 spill，但寄存器从 v3 的 31 增到 v4 的 40，再到 v5 的 64。按 256 threads/block、sm_61 常见每 SM 2048 threads 和 65536 registers 估算，v2/v3 主要受线程数限制，可到约 8 blocks/SM；v4 可能受寄存器限制降到约 6 blocks/SM；v5 进一步降到约 4 blocks/SM。这个 occupancy 判断是资源估算，不是 profiler 实测。

可验证的 NCU metric 名继续沿用 v3 已查询确认的指标：`smsp__inst_executed_op_shared_ld`、`smsp__inst_executed_op_shared_st`、`smsp__warp_issue_stalled_barrier_per_warp_active`、`dram__bytes_read`、`dram__bytes_write` 和 `sm__throughput`。本机仍未采集 Nsight Compute profiler 数据。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm -lcublas && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。v5 本轮重复运行 4 次，下表取每版最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4468 | 241.70 | 0.0604 | 0.000034 |
| v2_smem | 2.0413 | 33.39 | 0.1315 | 0.000034 |
| v3_reg_tile | 1.2530 | 41.01 | 0.2142 | 0.000034 |
| v4_2d_reg_tile | 0.8855 | 39.08 | 0.3031 | 0.000034 |

`GB/s` 对 v4 不能直接和 v3 当成“带宽利用率”比较，因为分母的有效访存字节不同。更直接的结论是：v4 比 v3 快约 `1.2530 / 0.8855 = 1.42x`，比 v2 快约 `2.31x`，比 naive 快约 `5.02x`。

### 当前瓶颈
v4 的主要限制已经不只是 DRAM 访存：

- 每个 `tile_k` 阶段仍有两次 `__syncthreads()`，同步次数没有减少。
- 每个线程维护 4 个 accumulator，register pressure 明显上升。
- 每个 K tile 需要搬 512 个 `A` 和 512 个 `B` 到 shared memory，比 v3 的 `A 512 + B 256` 更多。
- 仍然是标量 global load，没有 vectorized load，也没有 double buffering 来隐藏 load 延迟。

### 代价或限制
更大的 register tile 提高了算术强度，但不是无条件更快。继续扩大到 `4 x 4` 之类的形态时，寄存器可能导致 occupancy 下降、spill 或调度压力增加；shared memory tile 也会变大，加载分工和 bank conflict 风险都需要重新检查。

### 下一步
用户要求压缩到最后版本，因此下一节直接进入当前硬件可验证的收官版 v5。`vectorized load`、double buffering、`cp.async` 和 Tensor Core 不再拆成独立学习轮次；它们会在 v5 中作为“为什么当前没有做”的边界说明。

### v4 作业
1. 推导题：对 `32 x 32` 输出 tile，推导 v4 单个 tile 对 `A/B` 的 global load 数量，并把它摊到每个 `C` 元素。和 v3 相比，哪一项减少了？
2. 预测题：如果继续扩大到 `4 x 4` register tile，你预期有效 DRAM 访存、register pressure、occupancy 会分别怎样变化？为什么它不一定继续加速？

## v5 — final 4x4 register tile

### 本版学习目标
本轮按用户要求直接压缩到当前 GEMM 主线的收官版。本版唯一新增核心概念是 **当前硬件可验证的收官边界**：在 `sm_61` MX250 上继续扩大 register tile 到 `4 x 4`，并明确哪些更先进手段因为硬件或 dtype 前提不满足而不纳入可运行代码。

这里的“最后版本”不是业界最快 GEMM。业界最快形态通常依赖 Tensor Core、`mma`/`wmma`/`wgmma`、`cp.async`、更复杂的 pipeline 和特定 dtype/layout。当前项目默认是 FP32 CUDA C kernel，机器是 `compute capability 6.1`，所以本版选择一个可以真实编译、运行、benchmark 的收官版本，而不是写无法验证的伪代码。

### 硬件边界
本机 `nvidia-smi` 显示 GPU 为 NVIDIA GeForce MX250，compute capability 是 `6.1`。这带来几个直接限制：

- `cp.async` 是 Ampere 以后面向 shared memory 异步拷贝的能力，通常要求 `sm_80+`，当前 `sm_61` 不能作为本项目可运行版本。
- Tensor Core 从 Volta/Turing/Ampere 等后续架构开始进入主线，且通常要求 FP16/BF16/TF32/INT8 等 dtype 或特定 MMA 指令路径；当前版本仍是 FP32 scalar CUDA kernel。
- `TMA / wgmma / DSMEM / thread block cluster` 属于更晚架构的特性，不适合在 MX250 上作为默认教学代码。

所以 v5 的定位是：**当前硬件上的 FP32 手写 CUDA 收官版**，不是跨硬件的终极 GEMM。

### 改了什么
v5 保持 `blockDim = (16, 16)`，但一个 block 负责 `64 x 64` 输出 tile：

- `row_base = blockIdx.y * 64 + threadIdx.y`。
- `col_base = blockIdx.x * 64 + threadIdx.x`。
- 每个线程计算 4 行 x 4 列，共 16 个 `C` 元素。
- 每个线程维护 `sum[4][4]`，共 16 个寄存器累加器。
- shared memory 变为 `tile_a[64][16]` 和 `tile_b[16][64]`。

加载阶段中，每个线程加载 4 个 `A` 和 4 个 `B`：

```cpp
for i in 0..3:
    tile_a[ty + i * 16][tx] = A[row_base + i * 16, tile_k + tx]

for j in 0..3:
    tile_b[ty][tx + j * 16] = B[tile_k + ty, col_base + j * 16]
```

计算阶段中，每个 `inner` 读 4 个 `A` 值和 4 个 `B` 值，形成 `4 x 4 = 16` 次乘加：

```cpp
sum[i][j] += a_vals[i] * b_vals[j]
```

### 为什么可能更快
v4 的输出 tile 是 `32 x 32`，每个 `A` 元素服务 32 个输出列，每个 `B` 元素服务 32 个输出行。v5 把输出 tile 扩到 `64 x 64`，让两侧复用都变成 64。

按单个输出元素摊销：

- v4：`A = K/32`，`B = K/32`，写 `C = 1`。
- v5：`A = K/64`，`B = K/64`，写 `C = 1`。

这说明 v5 的有效 DRAM 访存继续下降。但收益不是免费的：每个线程的 accumulator 从 4 个增加到 16 个，register pressure 明显上升；每个 block 的 shared memory 从 4096B 增到 8192B；同一个 SM 可同时驻留的 block 数可能下降。

### 代码要点
- v5 使用小的固定循环和 `#pragma unroll`，让 `sum[4][4]`、`a_vals[4]`、`b_vals[4]` 尽量落在 registers。
- 所有 global load 都保留 `row < m`、`col < n`、`a_col < k`、`b_row < k` 边界检查；越界位置写 `0.0f` 到 shared memory。
- 不能在越界时提前 `return`，因为 block 后面仍有 `__syncthreads()`。
- 写回阶段对 16 个输出分别做边界检查，因此仍支持非整除的 `M/N/K`。

### 定量分析
v5 的 FLOPs 仍是 `M * N * 2K`。变化只在有效 global memory 访存：

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `(M*K*ceil(N/64) + K*N*ceil(M/64) + M*N) * 4` | `A` 对每个 64 列输出 tile 读一次，`B` 对每个 64 行输出 tile 读一次，`C` 写一次 |
| FLOPs `F` | `M * N * 2K` | 与前面版本相同 |
| 算术强度 `AI` | `F / B` | 默认规模约 `15.06 FLOP/Byte` |

默认 `M=N=K=512` 时：

| 项 | v4 | v5 | 变化 |
| --- | ---: | ---: | --- |
| `A` 有效读取 | 16 MiB | 8 MiB | 输出 tile 宽度 32 -> 64，减半 |
| `B` 有效读取 | 16 MiB | 8 MiB | 输出 tile 高度 32 -> 64，减半 |
| `C` 写回 | 1 MiB | 1 MiB | 输出元素数不变 |
| 总有效访存 | 34.60 MB | 17.83 MB | 约 `1.94x` 下降 |

瓶颈定性判定：**higher arithmetic intensity, register-limited risk**。v5 的有效访存约为 v4 的一半，但实测时间从 `0.8855 ms` 降到 `0.5058 ms`，约 `1.75x`，没有达到有效访存下降的 `1.94x`。这说明 v5 仍受益于更大 tile，但新增寄存器、shared memory 读写、指令调度和较低 occupancy 已经吃掉一部分收益。

ptxas 资源：

| kernel | registers/thread | shared memory/block | spill |
| --- | ---: | ---: | ---: |
| `kernel_v2` | 27 | 2048B | 0B |
| `kernel_v3` | 31 | 3072B | 0B |
| `kernel_v4` | 40 | 4096B | 0B |
| `kernel_v5` | 64 | 8192B | 0B |

按 `256 threads/block` 和 `65536 registers/SM` 粗略估算，v5 每个 block 使用 `256 * 64 = 16384` 个 registers，因此寄存器最多允许约 4 blocks/SM；线程数上限本来可允许 8 blocks/SM。这个估算说明 v5 很可能从 v4 的更高 occupancy 下降到更低 occupancy，但没有 spill，所以当前收益仍然大于代价。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm -lcublas && ./debugger/gemm
```

当前记录，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。本轮重复运行 4 次，下表取每版最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 4.4468 | 241.70 | 0.0604 | 0.000034 |
| v2_smem | 2.0413 | 33.39 | 0.1315 | 0.000034 |
| v3_reg_tile | 1.2530 | 41.01 | 0.2142 | 0.000034 |
| v4_2d_reg_tile | 0.8855 | 39.08 | 0.3031 | 0.000034 |
| v5_final_4x4 | 0.5058 | 35.24 | 0.5307 | 0.000034 |

相对当前实测最小值，v5：

- 比 v4 快约 `0.8855 / 0.5058 = 1.75x`。
- 比 v3 快约 `1.2530 / 0.5058 = 2.48x`。
- 比 v2 快约 `2.0413 / 0.5058 = 4.04x`。
- 比 naive 快约 `4.4468 / 0.5058 = 8.79x`。

注意 `GB/s` 数字随版本下降，不代表实际变慢。这里的 `GB/s` 用的是各版本自己的有效访存字节作为分母；v5 的分母已经只有约 `17.83 MB`，所以跨版本更应该先看 `ms` 和 `TFLOPS`。

### 当前瓶颈
v5 已经把有效 DRAM 访存压得比较低，下一层瓶颈更可能来自：

- register pressure：64 registers/thread 降低可驻留 block 数。
- shared memory 指令量：每个 `inner` 仍要从 shared memory 读 4 个 `A` 和 4 个 `B`。
- 同步：每个 `tile_k` 仍有两次 `__syncthreads()`。
- 标量 global load：代码没有使用 `float4` 或更复杂的协同 vectorized load。
- 没有 pipeline：load 和 compute 没有 double buffering 重叠。

### 后续硬件路径
如果目标是现代 NVIDIA GPU 上的高性能 GEMM，最后形态通常会转向 Tensor Core。那已经不是这个 FP32 scalar kernel 的连续小改动，而是 dtype、tile shape、fragment layout、warp-level MMA 指令和 pipeline 的整体重写。

为保持当前 `gemm.cu` 简洁，本文件只保留 v1-v5 FP32 手写路径和 `cublas_sgemm` 对比基线。Tensor Core / `cp.async` 后续可以单独开新文件或新学习轮次，而不是继续塞进同一个教学文件。

### v5 作业
1. 综合推导：从 v2、v3、v4 到 v5，分别写出每个版本每个 `C` 元素摊销的 `A/B` global load：`K/16`、`K/32`、`K/64` 分别来自哪里？
2. 预测题：如果把 v5 继续扩大到 `8 x 8` register tile，你预计会发生什么？请同时从有效 DRAM 访存、register pressure、occupancy、spill 风险和 block 数量五个角度判断。

## cuBLAS SGEMM 对比基线

### 本版学习目标
本轮新增的是 **library baseline**：用 cuBLAS 的 FP32 `cublasSgemm` 给手写 kernel 一个同机同规模的性能参考。它不是手写优化版本，也不解释 cuBLAS 内部如何 tiling；它的价值是回答“当前手写版本距离成熟库函数还有多远”。

### 改了什么
`src/gemm.cu` 新增：

- `#include <cublas_v2.h>` 和 `CUBLAS_CHECK(...)`。
- `cublasCreate(...)` / `cublasDestroy(...)` 管理 handle。
- `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`，让 SGEMM 使用更严格的 FP32 口径，避免默认 TF32 路径改变误差和比较对象。
- `launch_cublas(...)` 放进和 v1-v5 相同的 benchmark 版本列表。

### row-major 到 cuBLAS column-major
项目中 `A/B/C` 都是 row-major。cuBLAS 默认按 column-major 解释内存。

row-major 的 `C = A * B` 可以等价看成 column-major 的：

```text
C^T = B^T * A^T
```

所以代码不做额外转置，而是这样调用：

```cpp
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            b, n,
            a, k,
            &beta,
            c, n);
```

这里 `b` 在内存中被 cuBLAS 解释为 column-major 的 `B^T[N, K]`，`a` 被解释为 column-major 的 `A^T[K, M]`，输出 `c` 被解释为 column-major 的 `C^T[N, M]`，对应同一块 row-major `C[M, N]` 内存。

### 定量分析
cuBLAS 内部 tile、cache 和 kernel 选择是库实现细节，代码里不能像 v1-v5 那样精确写出每个 tile 的有效 global load。因此本轮 `print_row` 对 `cublas_sgemm` 使用最小有效访存口径：

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `(M*K + K*N + M*N) * 4` | 至少读一次 `A/B` 并写一次 `C`；实际 DRAM traffic 需 profiler 验证 |
| FLOPs `F` | `M * N * 2K` | 与 v1-v5 相同 |
| 算术强度 `AI` | `F / B` | 默认规模约 `85.33 FLOP/Byte` |

默认 `M=N=K=512` 时，最小有效访存是 `3,145,728 bytes`，FLOPs 是 `268,435,456`。这里的 `GB/s` 不能和 v1-v5 直接当成同一类带宽利用率比较；判断 cuBLAS 基线时优先看 `ms` 和 `TFLOPS`。

瓶颈定性判定：**compute-bound / small-GEMM overhead**。当前规模只有 `512^3`，cuBLAS 已明显快于手写 v5，但仍可能受问题规模较小、kernel launch、库内部 kernel 选择和 FP32 pedantic math 影响，不能代表大矩阵峰值。

可验证的 NCU metric 名沿用已有全局访存和 SM 指标：`dram__bytes_read`、`dram__bytes_write`、`dram__throughput`、`sm__throughput`。本轮没有采集 Nsight Compute profiler 数据。

### 实测结果
用下面命令编译并实测：

```bash
nvcc src/gemm.cu -o debugger/gemm -lcublas && ./debugger/gemm
```

当前记录，运行设备为 NVIDIA GeForce RTX 4060 Ti，compute capability `8.9`，默认 `M = 512, N = 512, K = 512`，`timeit` 使用 `warmup=3, iters=20`。本轮连续运行 3 次，下表取每版最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 0.5274 | 2037.86 | 0.5090 | 0.000034 |
| v2_smem | 0.4087 | 166.75 | 0.6568 | 0.000034 |
| v3_reg_tile | 0.2301 | 223.25 | 1.1664 | 0.000034 |
| v4_2d_reg_tile | 0.2072 | 167.00 | 1.2955 | 0.000034 |
| v5_final_4x4 | 0.1054 | 169.17 | 2.5476 | 0.000034 |
| cublas_sgemm | 0.0583 | 53.94 | 4.6031 | 0.000018 |

本轮 `cublas_sgemm` 比 v5 快约 `0.1054 / 0.0583 = 1.81x`。它的 `max_err` 更小，说明 row-major 映射和 FP32 SGEMM 口径正确通过当前 CPU reference。

### 代价或限制
- 现在编译 `src/gemm.cu` 需要链接 `-lcublas`。
- cuBLAS 基线不是教学 kernel，不能从源码直接看到 tile shape、register tile、pipeline 或 epilogue。
- `CUBLAS_PEDANTIC_MATH` 让比较更接近 FP32 手写 kernel；如果改用默认或 TF32 math mode，速度和误差都可能变化。
- `cublas_sgemm` 的 `GB/s` 使用最小有效访存字节，不能和 v1-v5 的手写逻辑访存口径直接比较。

### cuBLAS 基线作业
1. 解释题：为什么 row-major `C = A * B` 可以用 cuBLAS column-major 的 `C^T = B^T * A^T` 实现？请把 `m/n/k/lda/ldb/ldc` 分别对应到代码里的参数。
2. 判断题：为什么 `cublas_sgemm` 这一行的 `GB/s` 不适合直接和 v5 的 `GB/s` 比较？更应该优先比较哪个指标？

## 对比总表

本表中 v1-v5 和 `cublas_sgemm` 使用本轮 RTX 4060 Ti 实测的 3 次最小值。

| version | 核心手段 | 逻辑读写 | ms | GB/s | TFLOPS | max_err | 瓶颈 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程计算一个 `C[row, col]` | 每个输出读 `A/B` 各 `K` 次，写 `C` 1 次 | 0.5274 | 2037.86 | 0.5090 | 0.000034 | memory-bound + no explicit data reuse |
| v2_smem | shared memory tile | `A/B` 按 `16 x 16` 输出 tile 复用，默认有效访存约 `68.16 MB` | 0.4087 | 166.75 | 0.6568 | 0.000034 | less memory-bound + sync/shared-memory overhead |
| v3_reg_tile | `2 x 1` register tile | `B` 按 `32 x 16` 输出 tile 复用，默认有效访存约 `51.38 MB` | 0.2301 | 223.25 | 1.1664 | 0.000034 | less memory-bound + register/shared-memory reuse, still sync-limited |
| v4_2d_reg_tile | `2 x 2` register tile | `A/B` 按 `32 x 32` 输出 tile 复用，默认有效访存约 `34.60 MB` | 0.2072 | 167.00 | 1.2955 | 0.000034 | higher AI + higher register/shared-memory pressure |
| v5_final_4x4 | `4 x 4` register tile | `A/B` 按 `64 x 64` 输出 tile 复用，默认有效访存约 `17.83 MB` | 0.1054 | 169.17 | 2.5476 | 0.000034 | highest AI here + register-limited risk |
| cublas_sgemm | cuBLAS FP32 SGEMM | 最小有效 `A+B+C` 字节；内部 tiling 不透明 | 0.0583 | 53.94 | 4.6031 | 0.000018 | library baseline + small-GEMM overhead |

## 参考资料
- CUDA C++ Programming Guide：thread hierarchy、global memory coalescing、shared memory。
- cuBLAS Library：`cublasSgemm` column-major GEMM 调用约定。
- NVIDIA Nsight Compute CLI：后续 profiler 验证使用 `ncu` metric。
