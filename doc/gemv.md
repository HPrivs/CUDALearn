# GEMV

## 学习目标
- 写出正确的矩阵向量乘 CUDA kernel。
- 把 GEMV 理解成“每一行做一次 dot product”，也就是多组 reduce。
- 建立 naive 版本的 correctness、benchmark 和定量分析基线。
- 区分有效访存字节、FLOPs 和实际缓存行为。

## 前置知识
- row-major：二维数组按行连续存储，`A[row, col]` 的线性下标是 `row * cols + col`。
- dot product：两个等长向量对应元素相乘再求和。
- GEMV：general matrix-vector multiplication，矩阵向量乘，常见形式是 `y = A * x`。

## 问题规格
- 输入：矩阵 `A`，形状为 `M x K`
- 输入：向量 `x`，长度为 `K`
- 输出：向量 `y`，长度为 `M`
- dtype：`float32`
- 数学定义：`y[row] = sum(A[row, col] * x[col])`
- 默认规模：`M = 4096, K = 4096`
- 存储布局：`A` 为 row-major，`x/y` 为连续 1D 数组

GEMV 可以看成 `M` 个独立的 reduce：每个输出元素 `y[row]` 都需要把 `K` 个乘积累加起来。它比单个 reduce 多了行维度，也比 GEMM 少了输出列维度，适合用来过渡到二维 tiling。

## v1 — naive

### 本版学习目标
先写出最小正确实现：一个线程负责一个输出行，在线程内部串行完成这一行的 dot product。

### 改了什么
首版只做最直接的映射：

- `row = blockIdx.x * blockDim.x + threadIdx.x`
- 每个有效线程计算一个 `y[row]`
- 线程内部循环 `col = 0..K-1`，累加 `A[row * K + col] * x[col]`

### 为什么可能更快
这是 correctness 基线，不追求性能。它的价值是把 GEMV 的数学定义、行索引、CPU reference 和 benchmark 跑通。

### 代码要点
- `row < rows` 的边界检查必须保留，不能假设 `M` 一定能整除 `blockDim.x`。
- `A[row * cols + col]` 对单个线程来说是连续读取。
- 所有行都会反复读取同一个 `x[col]`，naive 版只依赖硬件 cache，不显式复用 `x`。
- `cpu_ref(...)` 用 `double` 做中间累加，再转回 `float`，降低 CPU 参考值自身的舍入误差。

### 定量分析
按 naive kernel 的逻辑访存计算，每个输出行需要读一行 `A`、读一遍 `x`、写一个 `y`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 + M * K * 4 + M * 4` | 读 `A`，每行读一次 `x`，写 `y` |
| FLOPs `F` | `M * (2K - 1)` | 每行 `K` 次乘法和 `K - 1` 次加法 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 默认大规模下，按逻辑读 `x` 计算 |

这里的 `B` 是有效逻辑字节，不等于真实 DRAM 字节。实际硬件可能把 `x` 缓存在 L2 或 L1 中，所以 `x` 不一定每一行都从 DRAM 重新读取一次。但 naive 版没有显式控制这种复用，因此分析时先把它当作需要验证的现象，而不是默认收益。

瓶颈定性判定：**memory-bound + low parallelism per row**。`AI` 很低，说明每读入 1 Byte 数据只做很少计算；同时一个线程串行完成整行 dot product，行内没有并行归约，`K` 很大时单线程工作过长。

可验证的 NCU metric 名：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`

这些 metric 名已用 `ncu --chips ga100 --metrics ... --query-metrics-mode suffix` 查询过。当前机器上直接 `ncu --query-metrics` 返回 `Skipping unsupported chip GP108`，所以本轮没有 profiler 实测。后续在 Nsight Compute 支持的 GPU 上验证时，重点比较 DRAM 吞吐、SM 吞吐以及 `FADD/FFMA` 指令数量。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/gemv.cu -o debugger/gemv && ./debugger/gemv
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 5.8505 | 22.94 | 0.0057 | 0.000168 |

本次 `max_err = 0.000168`，来自 CPU double 累加和 GPU float 累加的舍入差异，数量级合理。当前结果说明索引映射和边界处理通过了首轮 correctness 检查。

### 当前瓶颈
naive 版有两个明显限制：

- 行内串行：一个线程要完成 `K` 次乘加，不能利用一个 block 内多线程并行做同一行的 reduce。
- `x` 复用不显式：每一行都会访问同一个 `x`，但当前版本只依赖 cache 命中，不能保证所有访问都高效。

默认规模下，`A` 的读取量很大，`x` 也被逻辑上重复读取 `M` 次。因此这个版本预计主要受访存和行内串行累加限制。

### 代价或限制
代码最简单，没有 shared memory、没有同步、没有原子操作；代价是行内并行度为 1。`K` 越大，单线程串行循环越长，越不适合作为高性能实现。

### 下一步
下一版让一个 block 负责一行，用多个线程共同计算同一行的 dot product，再在 block 内归约 partial sum。目标是把行内串行 reduce 改成 block 内并行 reduce。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、边界检查、循环 `col` 完成 dot product、写回 `y[row]`。
2. 用一段话解释：为什么 GEMV naive 版的 `AI` 约为 `0.25 FLOP/Byte`，但实际 DRAM 访存可能小于文档里的逻辑 `B`？

## v2 — block-per-row shared memory reduce

### 本版学习目标
把 naive 版“一个线程串行算一整行”改成“一个 block 协作算一整行”，并用 shared memory 做 block 内归约。

block-per-row：每个 thread block 负责一个输出行 `y[row]`。block 内多个线程分别计算这一行的一部分乘加，最后合并成一个结果。

shared memory 是同一个 block 内线程共享的片上存储。本版用它保存每个线程的 partial sum，让 block 内线程能做树形归约。

### 改了什么
`v2` 的映射方式变成：

- `row = blockIdx.x`，一个 block 对应一行。
- `threadIdx.x` 从当前行中按 `col += blockDim.x` 读取多个列元素。
- 每个线程先在 register 里得到自己的 `sum`。
- 所有线程把 `sum` 写入 `smem[tid]`，再用 shared memory 树形归约。
- `tid == 0` 的线程把 `smem[0]` 写回 `y[row]`。

### 为什么可能更快
naive 版每行只有 1 个线程做 `K` 次乘加；默认 `K = 4096` 时，这条线程的串行循环很长。

`v2` 让 `256` 个线程共同处理一行。默认规模下，每个线程先处理约 `4096 / 256 = 16` 个列元素，然后 block 内做 `log2(256) = 8` 轮归约。它没有减少数学上的总访存字节和 FLOPs，但显著提高了行内并行度。

对固定的循环轮次 `col = tid + i * blockDim.x`，同一个 warp 中相邻线程访问相邻的 `A[row, col]` 和 `x[col]`，所以读取方向仍然是 coalesced access。

### 代码要点
- `kernel_v2<<<rows, kBlockSize>>>`：grid 的 x 维就是输出行数。
- `for (int col = tid; col < cols; col += blockDim.x)` 处理 `K` 不能整除 `blockDim.x` 的情况。
- 第一次 `__syncthreads()` 保证所有线程都已经写入 `smem[tid]`。
- 每轮归约后也要同步，否则下一轮可能读到尚未更新完成的 shared memory。
- `tid == 0` 写回 `y[row]`，没有跨 block 写同一个输出地址，因此不需要 `atomicAdd`。

### 定量分析
按有效逻辑字节计算，`v2` 和 naive 一样：每个输出行读一行 `A`、逻辑上读一遍 `x`、写一个 `y`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 + M * K * 4 + M * 4` | 读 `A`，每行逻辑读取一次 `x`，写 `y` |
| FLOPs `F` | `M * (2K - 1)` | 按 GEMV 数学定义统计有效乘加 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 有效口径下与 naive 基本相同 |

本版的收益不来自 `AI` 变大，而来自并行度变化：

| 项目 | naive | v2 |
| --- | ---: | ---: |
| 每行协作线程数 | 1 | 256 |
| 每线程主要读取元素数 | 4096 | 约 16 |
| 每行 block 内同步轮数 | 0 | 9 次左右 |
| 每行 global atomic | 0 | 0 |

瓶颈定性判定：**memory-bound + block reduce overhead**。`v2` 把行内串行累加大幅拆开后，主要限制会更接近输入读取带宽、`x` cache 复用效果，以及 shared memory 归约和 `__syncthreads()` 的开销。

可验证的 NCU metric 名：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`

当前机器上 `ncu --query-metrics` 返回 `Skipping unsupported chip GP108`，所以本轮没有 profiler 实测。后续在支持 Nsight Compute 的 GPU 上验证时，重点比较 `naive` 和 `v2` 的 DRAM 吞吐、SM 吞吐，以及 shared memory bank conflict 是否明显。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/gemv.cu -o debugger/gemv && ./debugger/gemv
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 5.8505 | 22.94 | 0.0057 | 0.000168 |
| v2_block_reduce | 1.4909 | 90.03 | 0.0225 | 0.000010 |

相对当前机器上的 `naive`，`v2_block_reduce` 约快 `3.92x`。这和预期一致：默认 `K = 4096` 时，naive 的单线程行内循环太长；`v2` 用 block 内并行归约减少了每个线程的串行工作量。

`v2` 的 `max_err` 比 naive 更小，不代表数学更精确的通用结论；它只是当前输入和当前归约顺序下，相对 CPU double reference 的最大差值更小。不同并行归约顺序会改变浮点舍入路径。

### 当前瓶颈
`v2` 已经解决了 naive 版最明显的行内串行问题，但还没有显式减少 `x` 的重复读取，也没有减少 block 内归约成本。

默认规模下，每个输出行启动一个 block。每个 block 都要读同一段 `x`，是否真的从 DRAM 重复读取取决于 cache；文档中的 `B` 仍是逻辑口径。block 内树形归约还需要 shared memory 读写和多次 `__syncthreads()`，这会在 `K` 较小或行数较多时更显眼。

### 代价或限制
- 每行需要一个 block，`M` 很小时可能 block 数不足，GPU 吃不满。
- `K` 很小时，block 内同步和 shared memory 归约开销可能超过并行累加收益。
- `blockDim.x = 256` 是当前教学配置，不一定适合所有 `K` 和 GPU。

### 下一步
下一版适合让一个 block 同时处理多行，也就是 **multi-row per block**。目标是当 `K` 不大或每行工作量不够时，提高 block 内线程利用率，并减少“一个 row 一个 block”带来的调度粒度限制。

### v2 作业
1. 概念题：为什么 `v2` 的 `AI` 和 naive 几乎相同，但实测速度能明显变快？请用“每行协作线程数”和“每线程串行循环长度”解释。
2. 改错题：下面代码少了两个关键点，请指出问题和可能后果。

```cpp
__global__ void kernel_v2_bug(const float* a, const float* x, float* y,
                              int rows, int cols) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        sum += a[row * cols + col] * x[col];
    }
    smem[tid] = sum;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
    }
    if (tid == 0) {
        y[row] = smem[0];
    }
}
```

## 对比总表

| version | 核心方法 | ms | GB/s | TFLOPS | max_err | 当前瓶颈 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程计算一行 | 5.8505 | 22.94 | 0.0057 | 0.000168 | memory-bound + 行内串行 |
| v2_block_reduce | 一个 block 计算一行，shared memory 归约 | 1.4909 | 90.03 | 0.0225 | 0.000010 | memory-bound + block reduce overhead |

## 参考资料
- 本项目 `src/gemv.cu`
- 本项目 `doc/reduce.md`
- 本项目 `doc/transpose.md`
