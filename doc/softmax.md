# Softmax

## 学习目标
- 写出正确的行级 softmax CUDA kernel。
- 理解 softmax 为什么通常先减去每行最大值。
- 建立 naive multi-pass 的 correctness、benchmark 和定量分析基线。
- 把 softmax 看成“行内 max reduce + 行内 sum reduce + elementwise 写回”的组合。
- 理解 online softmax 如何把 `max` 和 `sum` 统计合并到一次扫描。

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

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。本轮 v4 重新编译后连续跑两次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 104.8098 | 2.56 | 0.0006 | 0.000000 |

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

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。本轮 v4 重新编译后连续跑两次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 104.8098 | 2.56 | 0.0006 | 0.000000 |
| v2_block | 5.3348 | 50.32 | 0.0126 | 0.000000 |

相对当前机器上的 naive，v2 约快 `104.8098 / 5.3348 = 19.65x`。这符合预期：逻辑访存量没有变，但每行从单线程串行变成了 256 线程并行，行内工作长度明显下降。

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

## v3 — warp shuffle block reduce

### 本版学习目标
把 v2 的 shared memory tree reduce 改成 warp shuffle reduce。新增概念只有一个：用 `__shfl_down_sync` 在 warp 内直接交换寄存器值，减少 shared memory 读写和部分 block-level 同步。

warp shuffle：同一个 warp 内的线程直接读取彼此的 register 值，不需要先写到 shared memory。

### 改了什么
v3 保留 v2 的整体映射：

- 仍然是 `row = blockIdx.x`，一个 block 处理一行。
- 仍然每个线程以 `col += blockDim.x` 分摊列。
- 仍然三次扫描：求 `max`、求 `sum(exp)`、写回。

真正变化在 block reduce：

- 每个 warp 先用 `warp_reduce_max / warp_reduce_sum` 在 warp 内归约。
- 每个 warp 只让 lane 0 把该 warp 的局部结果写入 shared memory。
- 第一个 warp 再把这些 warp-level partial 归约成 block 级结果。
- v3 launch 不再需要动态 shared memory 参数：`kernel_v3<<<rows, kBlockSize>>>(...)`。

### 为什么可能更快
v2 的每一轮 tree reduce 都要读写 shared memory，并在每个 stride 后 `__syncthreads()`。`kBlockSize = 256` 时，一次 reduce 有 8 个 stride；max 和 sum 两次 reduce 会产生多轮同步。

v3 把 warp 内的 32 路归约放到 register shuffle 里完成。shared memory 只保存每个 warp 的 partial result，默认 256 线程只有 8 个 warp partial。理论上，v3 应该减少 shared memory load/store 指令和 barrier stall。

但 v3 没有改变三次 global memory 扫描，也没有减少两次 `expf`。如果主要时间花在 global memory 和 `expf` 上，减少 reduce 开销只能带来小幅收益。

### 代码要点
- `warp_reduce_max` 和 `warp_reduce_sum` 使用 `__shfl_down_sync(0xffffffff, val, offset)`，每次把距离为 `offset` 的 lane 的值拿过来合并。
- `block_reduce_max` 和 `block_reduce_sum` 先做 warp 内归约，再用 shared memory 合并 warp partial。
- `warp_vals[32]` 最多容纳 32 个 warp 的 partial，覆盖 CUDA 单 block 最多 1024 线程的情况。
- `tid < num_warps` 的线程负责读取 warp partial；其余线程用归约的单位元：max 用 `-FLT_MAX`，sum 用 `0.0f`。

### 定量分析
v3 的 global memory 逻辑访存和 v2 相同：每个输出元素读 `x` 三次、写 `y` 一次。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 * 4` | 三次读 `x`，一次写 `y`；不含 shared memory |
| FLOPs `F` | `M * K * 4` | 统计 2 次减法、1 次加法、1 次除法；不把比较、shuffle 和 `expf` 计入 FLOPs |
| 算术强度 `AI` | `F / B = 0.25 FLOP/Byte` | global memory 口径下不变 |

v3 改变的是片上归约开销。v2 的 shared memory tree reduce 对每个 block 做两次完整树形归约；v3 只把每个 warp 的 partial 写到 shared memory，再由第一个 warp 合并。这个改动主要应体现在 shared memory 指令数和 barrier stall 上，而不是 `dram__bytes_read/write` 上。

瓶颈定性判定：**latency-bound + expf/global-memory dominated**。v3 减少了 reduce 的片上开销，但当前默认规模下 reduce 不是唯一主耗时；三次 global load 和两次 `expf` 仍然保留，所以实测只小幅快于 v2。

可验证的 NCU metric 名：
- `smsp__inst_executed_op_shared_ld`
- `smsp__inst_executed_op_shared_st`
- `smsp__warp_issue_stalled_barrier_per_warp_active`
- `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `dram__bytes_read`
- `dram__bytes_write`

这些 metric 名已在本轮用 `ncu --chips ga100 --query-metrics` 查询确认。当前只写入 benchmark 数据，没有写入 profiler 实测数据；如果后续验证，重点看 v3 相比 v2 的 shared load/store 和 barrier stall 是否下降。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/softmax.cu -o debugger/softmax && ./debugger/softmax
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。本轮 v4 重新编译后连续跑两次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 104.8098 | 2.56 | 0.0006 | 0.000000 |
| v2_block | 5.3348 | 50.32 | 0.0126 | 0.000000 |
| v3_warp | 5.0719 | 52.93 | 0.0132 | 0.000000 |

相对 v2，v3 约快 `5.3348 / 5.0719 = 1.05x`，也就是约 `5.2%`。这个收益小于 v1 到 v2 的变化，说明当前版本的主要瓶颈已经不只是 shared memory reduce；warp shuffle 降低了片上归约成本，但没有改变 global memory 扫描和 `expf` 重复计算。

### 当前瓶颈
v3 剩下的主要问题：

- 仍然三次扫描 global memory。
- sum 阶段和 write 阶段仍然各算一次 `expf`。
- block 级结果仍需要跨 warp 合并，所以不能完全消除 shared memory 和 `__syncthreads()`。
- 对 `K = 4096` 这种较长行，reduce 开销占比有限；对更短的行，v3 的收益可能更明显，也可能被 launch/block 开销淹没，需要实测。

### 代价或限制
v3 代码比 v2 更复杂，需要理解 warp、lane、warp_id 和 shuffle mask。当前使用 `0xffffffff` 作为 mask，前提是参与 shuffle 的 warp lane 都是 active；本 kernel 每个 block 固定启动完整 warp，满足这个前提。

这个版本没有使用 online softmax，因此不能减少 pass 数，也不能减少第二次 `expf`。它只是优化了 block 内 reduce 的实现方式。

### 下一步
v4 已沿 online softmax 方向继续：把 max 和 sum 的统计合并到一次流式更新里，为后续 attention / FlashAttention 的分块 softmax 做准备。是否真的减少总时间，要看减少 pass 数和增加数学操作之间的取舍。

### v3 作业
1. 解释为什么 v3 只比 v2 小幅更快：请从 global memory 扫描、`expf`、shared memory reduce 三个角度回答。
2. 改错题：如果把 `block_reduce_sum` 里非 partial 线程的单位元写成 `-FLT_MAX`，会造成什么错误？为什么 max reduce 和 sum reduce 的单位元不同？

## v4 — online softmax

### 本版学习目标
把 v3 中分开的“先求 `max`，再求 `sum(exp(x - max))`”合并为一次 online 统计扫描。本轮新增概念只有一个：online softmax。

online softmax：扫描元素时维护当前最大值 `m` 和当前归一化分母 `d`，当最大值变大时，把旧的 `d` 按新最大值重新缩放。

### 改了什么
v4 继续复用 v3 的 block-per-row 映射和 warp shuffle block reduce：

- `row = blockIdx.x`，一个 block 处理一行。
- 每个线程仍然以 `col += blockDim.x` 分摊列。
- 每个线程不再分别计算 `thread_max` 和 `thread_sum`，而是维护一个 `OnlineState{max_val, denom}`。
- block 内归约不再只合并一个 float，而是合并两个状态 `(m, d)`。
- 第一轮扫描得到整行 `(max_val, denom)` 后，第二轮扫描写回 `y = exp(x - max_val) / denom`。

v3 对每行做三次 global memory 扫描：max pass、sum pass、write pass。v4 做两次扫描：online stats pass、write pass。

### 为什么可能更快
v4 减少了一次对 `x` 的 global memory 读取。按有效访存口径，每个输出元素从 v3 的 `3 read + 1 write = 16B` 变成 `2 read + 1 write = 12B`。

但这不是免费优化。online 更新在最大值变化时需要把旧分母乘上 `exp(old_max - new_max)`，block 内合并两个 partial state 时也需要按较大的 max 重新缩放另一个 partial 的 denom。因此 v4 减少了 DRAM 读取，但增加了少量 rescale 运算和 reduce 阶段的 `expf`。

### 代码要点
单个线程扫描自己负责的列时，状态更新分两种情况：

```cpp
if (x > m) {
    d = d * expf(m - x) + 1.0f;
    m = x;
} else {
    d += expf(x - m);
}
```

两个 partial state `(m_a, d_a)` 和 `(m_b, d_b)` 合并时，先取 `m = max(m_a, m_b)`，再把两个分母都缩放到同一个最大值基准：

```cpp
d = d_a * expf(m_a - m) + d_b * expf(m_b - m);
```

代码里为了少做一次 `expf(0)`，按 `a.max_val > b.max_val` 分支实现同一个公式。非 partial 线程的 online 单位元是 `OnlineState{-FLT_MAX, 0.0f}`。

### 定量分析
按 v4 online 的逻辑访存计算，每个输出元素读 `x` 两次、写 `y` 一次。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 3 * 4` | online stats 读一次 `x`，write pass 再读一次 `x` 并写一次 `y` |
| FLOPs `F` | `M * K * 4` | 按常见 update 路径统计：online update 的 1 次减法和 1 次加法，write pass 的 1 次减法和 1 次除法；不把比较、`expf`、rescale 乘法和 reduce 合并计入 FLOPs |
| 算术强度 `AI` | `F / B = 1 / 3 FLOP/Byte` | 有效字节减少，所以 AI 高于 v3 的 `0.25 FLOP/Byte` |

这个 `F` 仍然是简化口径。v4 的真实性能更依赖 `expf`、分支、warp reduce 中的状态合并，以及减少一次 global load 后是否真的降低了 DRAM 压力。

瓶颈定性判定：**latency-bound + online rescale/reduce overhead**。本轮实测 v4 快于 v3，说明减少一次 global memory 扫描在当前机器上有收益；但收益不是 `16B -> 12B` 的理想 1.33x，因为 online state 合并引入了额外数学操作，并且写回阶段仍然需要 `expf`。

可验证的 NCU metric 名沿用 v2/v3 中已查询确认的指标：
- `dram__bytes_read`
- `dram__bytes_write`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active`
- `smsp__warp_issue_stalled_barrier_per_warp_active`

如果后续用 NCU 验证，重点看 v4 相比 v3 的 `dram__bytes_read` 和 global load 指令数是否下降，同时观察 math pipe throttle 是否因为 online 合并增加。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/softmax.cu -o debugger/softmax && ./debugger/softmax
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。同一二进制连续跑了两次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 104.8098 | 2.56 | 0.0006 | 0.000000 |
| v2_block | 5.3348 | 50.32 | 0.0126 | 0.000000 |
| v3_warp | 5.0719 | 52.93 | 0.0132 | 0.000000 |
| v4_online | 4.5306 | 44.44 | 0.0148 | 0.000000 |

相对 v3，v4 约快 `5.0719 / 4.5306 = 1.12x`，也就是约 `11.9%`。这个结果说明 online softmax 在当前机器和默认规模下确实降低了总时间。

注意 v4 的 `GB/s` 数字低于 v3，不代表它更慢。这里的 `GB/s` 使用有效访存字节计算：v3 的分子是 `16B/elem`，v4 的分子是 `12B/elem`。当不同版本的有效字节不同，优先用 `ms` 和已说明的访存口径解释性能。

### 当前瓶颈
v4 已经把三次 global memory 扫描降到两次，但还剩几个限制：

- 写回阶段仍然需要再次读取 `x` 并计算 `expf(x - max_val)`。
- online state 合并比普通 sum reduce 更贵，因为 partial denom 需要按共同最大值重新缩放。
- 当前实现没有缓存整行 `exp` 或输入值，所以不能做到一次读完就直接写出。
- 对更短的 `K`，online 合并的额外开销可能占比更高；对更长的 `K`，少一次 global read 的收益可能更明显，需要实测。

### 代价或限制
v4 的数学推导比 v3 更复杂。它要求每个 partial state 的 `denom` 都明确是相对于自己的 `max_val` 计算的，合并时必须先统一 max 基准。漏掉缩放项会得到错误分母。

这个版本仍然没有做 vectorized load/store，也没有把 softmax 和后续矩阵乘融合。它适合作为 attention 分块 softmax 的前置概念，而不是最终高性能 softmax 终点。

### 下一步
Softmax 内部下一步可以做 vectorized load/store，观察 global load/store 指令数是否下降。另一条更有学习价值的路线是进入 Attention，把 online softmax 用在分块 `QK^T` 上，理解为什么不落地完整 score 矩阵能减少 HBM 往返。

### v4 作业
1. 概念题：给定旧状态 `(m_old, d_old)` 和新元素 `x`，分别写出 `x <= m_old` 与 `x > m_old` 两种情况下 online softmax 的更新公式，并解释为什么最大值变大时旧的 `d_old` 必须乘 `exp(m_old - x)`。
2. 改错题：如果合并两个 partial state 时直接写 `d = d_a + d_b`，但 `m_a != m_b`，会造成什么错误？写出正确的合并公式。

## 对比总表

| version | 核心方法 | ms | GB/s | TFLOPS | max_err | 当前瓶颈 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程处理一行，三次扫描 | 104.8098 | 2.56 | 0.0006 | 0.000000 | latency-bound + 行内串行 |
| v2_block | 一个 block 处理一行，shared memory 做 max/sum reduce | 5.3348 | 50.32 | 0.0126 | 0.000000 | latency-bound + reduction overhead |
| v3_warp | 一个 block 处理一行，warp shuffle 做大部分 reduce | 5.0719 | 52.93 | 0.0132 | 0.000000 | latency-bound + expf/global-memory dominated |
| v4_online | online 合并 max/sum，两次扫描 | 4.5306 | 44.44 | 0.0148 | 0.000000 | latency-bound + online rescale/reduce overhead |

v1-v3 的 `GB/s` 按 `16B/elem` 计算，v4 的 `GB/s` 按 `12B/elem` 计算；跨版本比较时先看 `ms`，再看对应的有效访存口径。

## 参考资料
