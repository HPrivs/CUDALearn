# Softmax

## 学习目标
- 写出正确的行级 softmax CUDA kernel。
- 理解 softmax 为什么通常先减去每行最大值。
- 建立 naive multi-pass 的 correctness、benchmark 和定量分析基线。
- 把 softmax 看成“行内 max reduce + 行内 sum reduce + elementwise 写回”的组合。

## 前置知识
- row-major：二维数组按行连续存储，`x[row, col]` 的线性下标是 `row * cols + col`。
- max reduce：从一组数中归约出最大值。
- softmax：把一行数转换成非负且总和为 1 的概率分布。

## 问题规格
- 输入：矩阵 `x`，形状为 `M x K`
- 输出：矩阵 `y`，形状为 `M x K`
- dtype：`float32`
- 数学定义：`y[row, col] = exp(x[row, col] - max(x[row, :])) / sum_j exp(x[row, j] - max(x[row, :]))`
- 默认规模：`M = 4096, K = 1024`
- 存储布局：`x/y` 都是 row-major

减去每行最大值不会改变 softmax 的数学结果，但能避免 `exp(x)` 在输入较大时溢出。这个版本先用最直接的三次行扫描建立基线，不提前引入 shared memory 或 warp reduce。

## v1 — naive multi-pass

### 本版学习目标
先写出最小正确实现：一个线程负责一整行，在这个线程内部完成 max、sum 和写回。

### 改了什么
首版只做最直接的映射：

- `row = blockIdx.x * blockDim.x + threadIdx.x`
- 每个有效线程处理一个完整行。
- 第 1 次扫描这一行，求 `max_val`。
- 第 2 次扫描这一行，累加 `denom = sum(exp(x - max_val))`。
- 第 3 次扫描这一行，写 `y = exp(x - max_val) / denom`。

### 为什么可能更快
这是 correctness 基线，不追求性能。它的价值是把 softmax 的数值稳定形式、行索引、CPU reference 和 benchmark 跑通。

### 代码要点
- `row < rows` 的边界检查必须保留，不能假设 `M` 一定能整除 `blockDim.x`。
- `max_val` 初始化为 `-FLT_MAX`，保证输入全为负数时也能得到正确最大值。
- CPU reference 用 `double` 计算 `max`、`exp` 和 `denom`，降低参考值自身的舍入误差。
- GPU kernel 用 `expf`，这是 `float32` 输入下的单精度指数函数。

### 定量分析
按 naive multi-pass 的逻辑访存计算，每个输出元素会读 `x` 三次、写 `y` 一次。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 * 4` | 三次读 `x`，一次写 `y` |
| FLOPs `F` | `M * K * 4` | 统计 2 次减法、1 次加法、1 次除法；不把比较和 `exp` 计入 FLOPs |
| 算术强度 `AI` | `F / B = 0.25 FLOP/Byte` | 这是本项目打印 `TFLOPS` 时使用的简化口径 |

这里的 `F` 不是完整代价模型。`expf` 是 special function，延迟和吞吐不能简单等同于 1 个普通 FLOP；max pass 的比较也没有计入 FLOPs。因此本版的 `TFLOPS` 只用于版本间趋势参考，不能拿去和 GEMM 的 `TFLOPS` 直接比较。

瓶颈定性判定：**latency-bound + low parallelism per row**。每行只有一个线程在串行执行 `K` 次比较、`2K` 次 `expf` 相关计算和多次 global load；即使逻辑 `AI` 很低，首要问题也不是只看 DRAM 带宽，而是行内并行度太低、special function 延迟难以隐藏。

可验证的 NCU metric 名：
- `dram__bytes`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `sm__sass_thread_inst_executed_op_fadd_pred_on`
- `sm__sass_thread_inst_executed_op_fmul_pred_on`

这些 metric 名已用 `ncu --chips ga100 --query-metrics` 查询。当前文档没有写入 profiler 实测数据；后续验证时重点看 DRAM 读写量、global load/store 指令数、SM 吞吐和 special function 相关 stall 现象。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/softmax.cu -o debugger/softmax && ./debugger/softmax
```

当前记录，默认 `M = 4096, K = 1024`，`timeit` 使用 `warmup=3, iters=20`：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 24.7529 | 2.71 | 0.0007 | 0.000000004 |

本次 `max_err ≈ 3.96e-09`，低于代码里的 `1e-5` correctness 阈值。`GB/s` 很低，说明 naive softmax 不是把 DRAM 带宽打满的形态；一个线程串行处理整行时，`expf` 延迟、循环串行和行内并行度不足更突出。

### 当前瓶颈
naive 版有三个明显限制：

- 行内串行：一个线程负责整行，不能利用一个 block 内多线程并行做 max reduce 和 sum reduce。
- 重复读取：同一个输入元素在 max、sum、write 三个阶段被逻辑读取三次。
- `expf` 延迟：每个元素至少在 sum 和 write 阶段各计算一次 `expf`，单线程串行时很难隐藏延迟。

默认规模下，这个版本预计主要受行内串行和 special function 延迟限制；global memory 访问也重要，但不是唯一瓶颈。

### 代价或限制
代码最简单，没有 shared memory、没有同步、没有跨线程归约；代价是每行并行度为 1，且每个输出元素对应三次输入读取。

这个版本适合作为正确性基线，不适合作为高性能实现。`K` 越大，单线程处理一整行的代价越高。

### 下一步
下一版让一个 block 负责一行，用多个线程共同求 `max` 和 `sum`，再并行写回整行输出。目标是把行内串行 reduce 改成 block 内并行 reduce。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、求 `max_val`、求 `denom`、写回 `y[row, col]`。
2. 用一段话解释：为什么 softmax 要先减去每行最大值？这个操作为什么不会改变 softmax 的结果？

## 对比总表

| version | 核心方法 | ms | GB/s | TFLOPS | max_err | 当前瓶颈 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程处理一行，三次扫描 | 24.7529 | 2.71 | 0.0007 | 0.000000004 | latency-bound + 行内串行 |

## 参考资料
