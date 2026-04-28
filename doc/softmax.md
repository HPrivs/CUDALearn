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
- 默认规模：`M = 4096, K = 4096`
- 存储布局：`x/y` 都是 row-major

减去每行最大值不会改变 softmax 的数学结果，但能避免 `exp(x)` 在输入较大时溢出。v1 先用最直接的三次行扫描建立基线，不提前引入 shared memory 或 warp reduce。

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

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 2.9928 | 89.69 | 0.0224 | 0.000000 |

本次 `max_err` 低于代码里的 `1e-5` correctness 阈值。`GB/s` 仍明显低于纯搬运类算子，说明 naive softmax 不是把 DRAM 带宽打满的形态；一个线程串行处理整行时，`expf` 延迟、循环串行和行内并行度不足更突出。

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

## v2 — block-per-row shared memory reduce

### 本版学习目标
把 v1 的“一个线程串行处理一整行”改成“一个 block 处理一行”。本轮只新增一个主要手段：用 shared memory 在 block 内做 `max` 和 `sum` 归约。

shared memory reduction：把每个线程的局部结果先写入 shared memory，再由同一个 block 内的线程逐步合并成一个结果。

### 改了什么
v2 的映射方式变成：

- `row = blockIdx.x`，一个 block 固定负责一行。
- `threadIdx.x` 以 `col += blockDim.x` 的方式分摊这一行的列。
- 每个线程先求自己负责列上的 `thread_max`。
- 把 `thread_max` 写入 `smem[tid]`，通过 shared memory reduce 得到整行 `max_val`。
- 再用同样方式求每个线程的 `thread_sum`，reduce 得到整行 `denom`。
- 最后每个线程写回自己负责的列。

### 为什么可能更快
v1 每行只有 1 个线程，`K = 4096` 时这个线程要串行完成整行的比较、两轮 `expf` 和写回。v2 让 256 个线程共同处理一行，单个线程大约只负责 `4096 / 256 = 16` 个元素，行内串行长度大幅下降。

这不是减少了逻辑访存量。v2 仍然按三次行扫描读取 `x`，再写一次 `y`；主要收益来自行内并行度提高，以及更多 warp 同时执行时更容易隐藏 `expf` 和 memory latency。

### 代码要点
- `kernel_v2<<<rows, kBlockSize, kBlockSize * sizeof(float)>>>(...)`：grid 的每个 block 对应一行，动态 shared memory 保存每个线程的局部归约值。
- `for (int col = tid; col < cols; col += blockDim.x)`：真实处理 `cols` 不能整除 `blockDim.x` 的情况。
- 两次 shared memory reduce 分别服务于 `max_val` 和 `denom`，每轮 reduce 后都要 `__syncthreads()`，否则有线程可能读到其他线程尚未写完的数据。
- 当前 `kBlockSize = 256`，是 2 的幂；代码里的二分 reduce 依赖这个前提。

### 定量分析
按逻辑访存计算，v2 和 v1 一样，每个输出元素读 `x` 三次、写 `y` 一次。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 * 4` | 三次读 `x`，一次写 `y` |
| FLOPs `F` | `M * K * 4` | 统计 2 次减法、1 次加法、1 次除法；不把比较、`fmaxf` 和 `expf` 计入 FLOPs |
| 算术强度 `AI` | `F / B = 0.25 FLOP/Byte` | 逻辑 AI 不变，所以 v2 的加速不是来自提高 AI |

默认规模下，`M * K = 16,777,216`，有效访存字节约 `268.4 MB`，有效 FLOPs 约 `67.1 MFLOPs`。v2 的 `GB/s` 和 `TFLOPS` 都按同一套有效口径计算，用于和 v1 相对比较。

瓶颈定性判定：**latency-bound + reduction overhead**。v2 已经修复 v1 最大的行内串行问题，但每行仍要三次扫 global memory，并且每行有两次 block-level reduction、两组 `__syncthreads()` 循环和两次 `expf` 计算。下一步如果继续优化，重点不只是减少 reduce 开销，还要考虑如何少读一次、少算一次 `expf` 或用 warp-level primitive 降低同步成本。

可验证的 NCU metric 名：
- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `smsp__warp_issue_stalled_barrier_per_warp_active`
- `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active`

这些 metric 名已在本轮用 `ncu --chips ga100 --query-metrics` 查询确认。当前只写入 benchmark 数据，没有写入 profiler 实测数据；如果后续用 NCU 验证，重点看 global load/store、barrier stall 和 math pipe throttle 是否符合上面的瓶颈判断。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/softmax.cu -o debugger/softmax && ./debugger/softmax
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。同一程序连续跑了三次，趋势一致；下表记录最终验证结果：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 2.9928 | 89.69 | 0.0224 | 0.000000 |
| v2_block | 0.6203 | 432.76 | 0.1082 | 0.000000 |

相对当前机器上的 naive，v2 约快 `2.9928 / 0.6203 = 4.82x`。这符合预期：逻辑访存量没有变，但每行从单线程串行变成了 256 线程并行，行内工作长度明显下降。

### 当前瓶颈
v2 的主要瓶颈已经不是“每行只有一个线程”。剩下的问题集中在：

- 三次 global memory 行扫描没有减少。
- sum 阶段和 write 阶段各算一次 `expf`，存在重复计算。
- shared memory reduce 每个 row 做两次，且每次都有多轮 `__syncthreads()`。
- 一个 block 只处理一行；当 `K` 很小的时候，256 个线程可能利用不满。

### 代价或限制
v2 比 v1 多用了 shared memory 和 block 内同步。对于 `K` 较大的行，这个代价通常能被行内并行收益覆盖；对于 `K` 很小的行，同步和 block 调度开销可能反而显得更重。

当前 reduce 写法假设 `blockDim.x` 是 2 的幂。默认 `kBlockSize = 256` 满足这个条件；如果后续把 block size 改成非 2 的幂，需要同步修改 reduce 逻辑。

### 下一步
下一版可以考虑 warp-level softmax：用 `__shfl_down_sync` 在 warp 内归约，减少 shared memory 读写和部分 `__syncthreads()`。另一条路线是 online softmax，用更接近 attention 的方式把 max 和 sum 的统计合并起来，为后续 FlashAttention 铺路。

### v2 作业
1. 解释 v2 为什么比 v1 快：请分别从“每行并行度”“逻辑访存量”“同步开销”三个角度回答。
2. 改错题：下面这段 reduce 少了一处关键同步。指出 bug 在哪里，并说明可能造成什么错误。

```cpp
smem[tid] = thread_max;
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
}
```

## 对比总表

| version | 核心方法 | ms | GB/s | TFLOPS | max_err | 当前瓶颈 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程处理一行，三次扫描 | 2.9928 | 89.69 | 0.0224 | 0.000000 | latency-bound + 行内串行 |
| v2_block | 一个 block 处理一行，shared memory 做 max/sum reduce | 0.6203 | 432.76 | 0.1082 | 0.000000 | latency-bound + reduction overhead |

## 参考资料
