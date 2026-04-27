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
| naive | 5.6205 | 23.88 | 0.0060 | 0.000168 |

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

## 对比总表

| version | 核心方法 | ms | GB/s | TFLOPS | max_err | 当前瓶颈 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程计算一行 | 5.6205 | 23.88 | 0.0060 | 0.000168 | memory-bound + 行内串行 |

## 参考资料
- 本项目 `src/gemv.cu`
- 本项目 `doc/reduce.md`
- 本项目 `doc/transpose.md`
