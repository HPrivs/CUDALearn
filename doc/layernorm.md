# LayerNorm / RMSNorm

## 学习目标
- 写出正确的行级 LayerNorm CUDA kernel。
- 理解 LayerNorm 的 mean、variance 和 normalize 三个阶段。
- 建立 naive multi-pass 的 correctness、benchmark 和定量分析基线。
- 明确哪些后续优化是复用旧技巧，哪些才是本算子的新学习重点。
- 用 Welford variance 把 mean 和 variance 的统计合并到一次扫描，并评估访存减少与额外计算之间的取舍。
- 实现无 affine RMSNorm，对比它和 LayerNorm 在统计量、访存量和归约开销上的差异。

## 前置知识
- row-major：二维数组按行连续存储，`x[row, col]` 的线性下标是 `row * cols + col`。
- mean：一组数的平均值，LayerNorm 中按每一行独立计算。
- variance：一组数围绕 mean 的平均平方偏差，反映这一行的离散程度。
- epsilon：加在 variance 上的小正数，用来避免除以 0 并改善数值稳定性。
- Welford variance：一种在线更新 mean 和平方偏差累积量 `M2` 的算法，可把多个 partial statistics 稳定合并。

## 问题规格
- 输入：矩阵 `x`，形状为 `M x K`
- 输出：矩阵 `y`，形状为 `M x K`
- dtype：`float32`
- 数学定义：`y[row, col] = (x[row, col] - mean(row)) / sqrt(var(row) + eps)`
- `mean(row) = sum_j x[row, j] / K`
- `var(row) = sum_j (x[row, j] - mean(row))^2 / K`
- RMSNorm 定义：`y[row, col] = x[row, col] / sqrt(mean_square(row) + eps)`
- `mean_square(row) = sum_j x[row, j]^2 / K`
- 默认规模：`M = 4096, K = 4096`
- 默认 `eps = 1e-5`
- 存储布局：`x/y` 都是 row-major

v1-v3 暂时不加入 `gamma/beta`，目标是把 LayerNorm 本身的三阶段公式、CPU reference 和 benchmark 跑通。v4 加入无 affine RMSNorm，用同一套 block-per-row 基础设施比较归一化算法变体。

## v1 — naive multi-pass

### 本版学习目标
先写出最小正确实现：一个线程负责一整行，在这个线程内部完成 mean、variance 和写回。

### 改了什么
首版只做最直接的映射：

- `row = blockIdx.x * blockDim.x + threadIdx.x`
- 每个有效线程处理一个完整行。
- 第 1 次扫描这一行，累加 `sum` 并计算 `mean`。
- 第 2 次扫描这一行，累加平方偏差 `sq_sum` 并计算 `var`。
- 第 3 次扫描这一行，写 `y = (x - mean) * inv_std`。

### 为什么可能更快
这是 correctness 基线，不追求性能。它的价值是把 LayerNorm 的统计量、行索引、CPU reference 和 benchmark 跑通。

和 Softmax v1 类似，naive LayerNorm 的问题不是公式复杂，而是把一整行都交给一个线程串行处理。后续 optimized baseline 会直接复用已学过的 block-per-row、shared memory 和 warp shuffle 技巧，不把这些旧技巧拆成多轮重复教学。

### 代码要点
- `row < rows` 的边界检查必须保留，不能假设 `M` 一定能整除 `blockDim.x`。
- CPU reference 用 `double` 计算 `mean`、`variance` 和 `sqrt`，降低参考值自身的舍入误差。
- GPU kernel 用 `float` 累加，和常见 `float32` kernel 的基础实现保持一致。
- `inv_std = 1.0f / sqrtf(var + eps)` 只在每行计算一次，写回阶段复用这个标量。

### 定量分析
按 naive multi-pass 的逻辑访存计算，每个输出元素会读 `x` 三次、写 `y` 一次。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 * 4` | 三次读 `x`，一次写 `y` |
| FLOPs `F` | `M * K * 6` | sum 阶段 1 次加法；variance 阶段 1 次减法、1 次乘法、1 次加法；write 阶段 1 次减法、1 次乘法 |
| 算术强度 `AI` | `F / B = 0.375 FLOP/Byte` | 不把 `sqrtf`、除法和每行级别的 scale 操作计入打印口径 |

默认规模下，`M * K = 16,777,216`，有效访存字节约 `268.4 MB`，有效 FLOPs 约 `100.7 MFLOPs`。这里的 `F` 是本项目用于版本间比较的简化口径，不是完整硬件代价模型。

瓶颈定性判定：**latency-bound + low row-level parallelism + uncoalesced global access**。每行只有一个线程串行扫描 `K` 个元素；同一个 warp 内的相邻线程处理不同行，在相同循环位置访问的地址相隔 `K * sizeof(float)`，不是连续合并访问。因此虽然逻辑 `AI` 很低，naive 版也很难打满 DRAM 带宽。

可验证的 NCU metric 名：
- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`

当前机器的 GP108 不被本机 `ncu --query-metrics` 支持；上述 metric 名按项目已有做法，用 `ncu --chips ga100 --query-metrics` 查询确认。本文档没有写入 profiler 实测数据。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/layernorm.cu -o debugger/layernorm && ./debugger/layernorm
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。加入 v4 后统一重复运行 3 次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 99.5988 | 2.70 | 0.0010 | 0.000003 |

本次 `max_err = 3.21865e-06`，低于代码里的 `1e-5` correctness 阈值。有效 `GB/s` 很低，说明 naive LayerNorm 没有把 DRAM 带宽打满；主要限制来自每行单线程串行扫描、warp 内跨行 stride 访问和较少的可并行行级工作。

### 当前瓶颈
naive 版有三个明显限制：

- 行内串行：一个线程负责整行，`K = 4096` 时每个线程要完成三次长循环。
- 访问不合并：同一个 warp 的线程在相同 `col` 上访问不同行，地址跨度是整行长度。
- 重复读取：同一个输入元素在 mean、variance 和 write 三个阶段被逻辑读取三次。

### 代价或限制
代码最简单，没有 shared memory、没有同步、没有跨线程归约；代价是每行并行度为 1，且 global memory 访问形态对 warp coalescing 不友好。

这个版本适合作为正确性基线，不适合作为高性能实现。`K` 越大，单线程处理一整行的串行代价越高。

### 下一步
v2 已经把 block-per-row、shared memory 和 `__shfl_down_sync` 合并成一个 optimized baseline，用 benchmark 说明这些旧技巧迁移到 LayerNorm 后的净效果。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、求 `mean`、求 `var`、计算 `inv_std`、写回 `y[row, col]`。
2. 用一段话解释：为什么 naive LayerNorm 的有效 `GB/s` 很低？请至少从“每行并行度”和“warp 内访存是否连续”两个角度回答。

## v2 — block-per-row + warp shuffle reduce

### 本版学习目标
本轮新增核心概念：把 LayerNorm 这种行归约算子从“一个线程处理一行”改成“一个 block 并行处理一行”。

复用旧技巧：`__shfl_down_sync` 和少量 shared memory 组成 `block_reduce_sum`。这些技巧已经在 Reduce / Softmax 中学过，本轮不重新展开，只关注它们迁移到 LayerNorm 后解决了什么瓶颈。

### 改了什么
v2 使用 `blockIdx.x` 对应一行，`threadIdx.x` 对应该行内的列方向工作：

- `row = blockIdx.x`，一个 block 处理一整行。
- 每个线程按 `col = tid; col < cols; col += blockDim.x` 扫描本行的一部分。
- 第一次扫描得到每个线程的 `thread_sum`，再用 `block_reduce_sum` 得到整行 `sum` 和 `mean`。
- 第二次扫描得到每个线程的 `thread_sq_sum`，再归约得到 `var`。
- 第三次扫描由所有线程并行写回 `y`。

### 为什么可能更快
v1 的 warp 内相邻线程处理相邻行，在相同 `col` 上访问地址相隔 `cols * sizeof(float)`。v2 中同一个 warp 的线程处理同一行的连续列，第一次迭代访问 `x[row_offset + tid]`，因此更接近连续合并访存。

同时，v1 每行只有一个线程串行做三次长循环；v2 每行有 `kBlockSize = 256` 个线程分摊行内扫描，并用 block 内归约合并 partial sum。默认 `cols = 4096` 时，每个线程每次扫描大约处理 16 个元素，行内串行长度大幅下降。

### 代码要点
- `kernel_v2` 的 `row` 必须是 `blockIdx.x`，不能再写成 `blockIdx.x * blockDim.x + threadIdx.x`，否则一个 block 内的线程会分散到不同行。
- `for (int col = tid; col < cols; col += blockDim.x)` 让同一 warp 的线程访问同一行的连续列。
- `block_reduce_sum` 先在 warp 内用 `__shfl_down_sync` 归约，再把每个 warp 的结果写到 shared memory，最后由 warp 0 归约这些 warp partial。
- v2 仍然保留三次 global memory 扫描；本轮优化的是行内并行度和访问形态，不是减少逻辑访存次数。

### 定量分析
v2 的逻辑访存量和 v1 相同：mean 阶段读一次 `x`，variance 阶段读一次 `x`，normalize 阶段读一次 `x` 并写一次 `y`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 4 * 4` | 三次读 `x`，一次写 `y` |
| FLOPs `F` | `M * K * 6` | 仍按 v1 的有效 FLOPs 口径统计 |
| 算术强度 `AI` | `F / B = 0.375 FLOP/Byte` | 逻辑 AI 没变，变化来自并行度和 memory coalescing |

默认规模下，`B` 约 `268.4 MB`，`F` 约 `100.7 MFLOPs`。因为 `AI` 仍然很低，v2 仍主要是 memory-bound；但相比 v1，它更能把 global memory 访问组织成连续访问，并且用更多线程隐藏 latency。

当前瓶颈：**memory-bound + block reduction / synchronization overhead**。v2 已经显著改善行内串行和 uncoalesced access，但每行仍要做两次 block 归约；当前 `block_reduce_sum` 每次包含两次 `__syncthreads()`，并且仍然三次读取 `x`。

可验证的 NCU metric 名沿用 v1 已列出的查询结果：
- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/layernorm.cu -o debugger/layernorm && ./debugger/layernorm
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。加入 v4 后统一重复运行 3 次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 99.5988 | 2.70 | 0.0010 | 0.000003 |
| v2_block_warp | 4.7923 | 56.01 | 0.0210 | 0.000000 |

v2 相对当前重新实测的 naive 约快 `20.8x`。这说明 v1 的主要损失不是 LayerNorm 公式本身，而是线程映射和访存形态：把同一行交给一个 block 后，行内扫描并行化，warp 内访问也从跨行 stride 变成行内连续。

### 代价或限制
v2 增加了两次 block 内归约和跨 warp shared memory 合并，代码复杂度高于 naive。对于很小的 `cols`，block-per-row 可能浪费线程；对于很大的 `cols`，仍然要三次读取 `x`，global memory traffic 还没有减少。

### 下一步
v3 转向 LayerNorm 自身的新问题：用 Welford variance 合并 mean 和 variance 的统计扫描。

### v2 作业
1. 概念题：为什么 v2 的逻辑访存字节 `B` 和 v1 一样，但实测 `GB/s` 和耗时明显改善？请从行内并行度和 memory coalescing 两个角度回答。
2. 改错题：下面这段 v2 风格代码有两个和 block-per-row 映射相关的问题，请指出并说明后果。

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const size_t row_offset = static_cast<size_t>(row) * cols;

float thread_sum = 0.0f;
for (int col = 0; col < cols; col += blockDim.x) {
    thread_sum += x[row_offset + col];
}
```

## v3 — Welford variance

### 本版学习目标
本轮新增核心概念：用 Welford 在线方差算法在一次扫描中同时得到每行的 `mean` 和 `variance` 统计量。

复用旧技巧：仍然使用 v2 的 block-per-row 映射和 warp shuffle 跨线程合并。区别是归约对象不再是单个 `float sum`，而是 `{mean, m2, count}` 这个统计结构。

### 改了什么
v2 的统计阶段需要两次读 `x`：

- 第一次读 `x` 求 `sum` 和 `mean`。
- 第二次读 `x` 用 `x - mean` 求 `sq_sum` 和 `var`。
- 第三次读 `x` 并写回 `y`。

v3 改成：

- 每个线程在第一次扫描中用 `welford_update` 得到自己的 partial `{mean, m2, count}`。
- `block_reduce_welford` 把同一行内所有线程的 partial statistics 合并成整行统计量。
- 第二次扫描读取 `x`，用最终 `mean` 和 `inv_std` 写回 `y`。

这里的 `m2` 表示平方偏差累积量，最终 population variance 是 `var = m2 / cols`。LayerNorm 默认使用除以 `K` 的方差，不是样本方差的 `K - 1`。

### 为什么可能更快
理论上，v3 把 global memory 逻辑访问从三次读 `x`、一次写 `y` 降到两次读 `x`、一次写 `y`。默认规模下，有效访存字节从约 `268.4 MB` 降到约 `201.3 MB`，少了 25%。

但 Welford 不是免费优化。每个元素更新统计量时要维护 `count`、`mean` 和 `m2`，并且跨线程合并时要同时 shuffle/合并三个字段。它减少的是 DRAM 逻辑读取，增加的是算术、寄存器和归约逻辑。

### 代码要点
- `WelfordData` 保存 `mean`、`m2`、`count`。
- `welford_update` 用一个新元素更新当前线程的 partial statistics。
- `welford_combine` 不能简单相加两个 mean；必须用两个 partial mean 的差值 `delta` 修正合并后的 `m2`。
- `block_reduce_welford` 的结构和 v2 的 block reduce 类似，但每次 shuffle 三个字段。
- v3 仍然真实处理 `cols` 非 `blockDim.x` 整除的情况：每个线程按 `col = tid; col < cols; col += blockDim.x` 扫描。

### 定量分析
v3 的逻辑访存量减少：统计阶段读一次 `x`，normalize 阶段读一次 `x` 并写一次 `y`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 3 * 4` | 两次读 `x`，一次写 `y` |
| FLOPs `F` | `M * K * 7` | Welford update 按 5 个简化算术操作计，write 阶段 2 个操作；不把除法和 `sqrtf` 计入打印口径 |
| 算术强度 `AI` | `F / B ≈ 0.583 FLOP/Byte` | 打印口径下比 v2 更高，但实际还多了 per-element division 和更复杂的 combine |

默认规模下，`B` 约 `201.3 MB`，`F` 约 `117.4 MFLOPs`。如果 kernel 主要受 DRAM 读取限制，v3 应该快于 v2；如果 Welford 的额外算术、寄存器或归约开销占主导，v3 可能只持平甚至变慢。

当前瓶颈：**memory-bound + Welford arithmetic/reduction overhead**。这轮的关键不是“Welford 一定更快”，而是学习如何判断减少一次 global read 是否足以抵消新增计算。

可验证的 NCU metric 名沿用前面已查询确认的指标：
- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`

本文档不新增未查询确认的 metric 名。如果后续做 profiler，重点比较 v2/v3 的 `dram__bytes_read` 是否下降，以及 Source 页面中 Welford update/combine 相关代码行是否成为新热点。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/layernorm.cu -o debugger/layernorm && ./debugger/layernorm
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。加入 v4 后统一重复运行 3 次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 99.5988 | 2.70 | 0.0010 | 0.000003 |
| v2_block_warp | 4.7923 | 56.01 | 0.0210 | 0.000000 |
| v3_welford | 4.2384 | 47.50 | 0.0277 | 0.000000 |

现象：v3 的正确性通过，当前重新实测下比 v2 耗时降低约 `11.6%`，速度约 `1.13x`。这说明当前机器和默认规模下，少读一次 `x` 有可见收益；但没有接近 25% 的逻辑访存减少幅度，因为 Welford 的逐元素更新、字段 shuffle 和 combine 开销吃掉了一部分收益。

注意这里的 `GB/s` 不能孤立解读：v3 的有效字节 `B` 比 v2 少 25%，所以即使 v3 的 `ms` 更低，按有效字节计算出来的 `GB/s` 仍可能更低。跨版本比较时先看 `ms`，再结合 `B` 和额外计算解释原因。

### 代价或限制
v3 的代码复杂度明显高于 v2，并且 Welford update 中有 per-element division。它的优势更可能出现在 global memory 读取更贵、`K` 更大、或者后续还能继续融合 affine/RMSNorm 写回逻辑的场景。

它没有减少 normalize 阶段的最后一次读 `x`，也没有加入 `gamma/beta`。如果下一步继续 LayerNorm，比较自然的方向是 fused affine；如果转向 RMSNorm，则可以直接少掉 mean 统计。

### v3 作业
1. 概念题：v3 的逻辑访存字节比 v2 少 25%，为什么实测耗时只降低约 11.6%，没有接近 25%？请从 DRAM 读减少、Welford per-element arithmetic、Welford combine 三个角度回答。
2. 改错题：下面的 Welford partial 合并方式有什么问题？为什么不能这样合并两个 partial statistics？

```cpp
WelfordData wrong_combine(WelfordData a, WelfordData b) {
    WelfordData out;
    out.count = a.count + b.count;
    out.mean = a.mean + b.mean;
    out.m2 = a.m2 + b.m2;
    return out;
}
```

## v4 — RMSNorm

### 本版学习目标
本轮新增核心概念：RMSNorm。RMSNorm 是 Root Mean Square Normalization，只用每行的均方根做缩放，不减去 mean。

复用旧技巧：仍然使用 v2 的 block-per-row 映射和 `block_reduce_sum`。本轮不新增 warp primitive，只观察去掉 mean 后统计阶段的计算和归约对象如何变化。

### 改了什么
v4 实现无 affine RMSNorm：

- 第一次扫描每行，累加 `sum(x^2)`。
- 用 `mean_square = sum(x^2) / cols` 得到均方。
- 计算 `inv_rms = 1 / sqrt(mean_square + eps)`。
- 第二次扫描写回 `y = x * inv_rms`。

代码中新增了 `cpu_ref_rmsnorm`，因为 RMSNorm 和 LayerNorm 数学定义不同，不能继续用 LayerNorm 的 CPU reference 做 correctness。benchmark 的 `Version` 条目也记录各自对应的 reference，v1-v3 对 LayerNorm reference，v4 对 RMSNorm reference。

### 为什么可能更快
和 v3 Welford LayerNorm 相比，v4 的逻辑 global memory traffic 一样，都是两次读 `x`、一次写 `y`。真正变化在统计量：

- v3 要维护 `{mean, m2, count}`，并在 warp/block reduce 中合并三个字段。
- v4 只归约一个 `sum(x^2)`，归约对象重新变回单个 `float`。
- v4 每个元素统计阶段只做 `val * val + add`，没有 Welford 的 per-element division 和复杂 combine。

这不是“更快的 LayerNorm”，而是一个不同的归一化算子。速度比较只能说明在相同输入形状和实现框架下，RMSNorm 的统计阶段更轻。

### 代码要点
- `thread_sq_sum += val * val`，不能只累加 `val`。
- `mean_square` 必须除以 `cols`，否则 `K` 越大，归一化尺度越错。
- RMSNorm 写回是 `x * inv_rms`，没有 `(x - mean)`。
- v4 继续使用 `for (int col = tid; col < cols; col += blockDim.x)`，所以非整除规模仍能覆盖尾部元素。

### 定量分析
v4 的逻辑访存量和 v3 一样：统计阶段读一次 `x`，normalize 阶段读一次 `x` 并写一次 `y`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `M * K * 3 * 4` | 两次读 `x`，一次写 `y` |
| FLOPs `F` | `M * K * 3` | 统计阶段 1 次乘法、1 次加法；write 阶段 1 次乘法 |
| 算术强度 `AI` | `F / B = 0.25 FLOP/Byte` | 不把每行一次的除法和 `sqrtf` 计入打印口径 |

默认规模下，`B` 约 `201.3 MB`，`F` 约 `50.3 MFLOPs`。`AI` 很低，v4 仍主要是 **memory-bound + block reduction overhead**；但它的 reduction 比 v3 轻，因为只归约一个 `float`。

可验证的 NCU metric 名沿用前面已查询确认的指标：
- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `sm__throughput`
- `smsp__inst_executed_op_global_ld`
- `smsp__inst_executed_op_global_st`
- `sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/layernorm.cu -o debugger/layernorm && ./debugger/layernorm
```

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。本轮重复运行 3 次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 99.5988 | 2.70 | 0.0010 | 0.000003 |
| v2_block_warp | 4.7923 | 56.01 | 0.0210 | 0.000000 |
| v3_welford | 4.2384 | 47.50 | 0.0277 | 0.000000 |
| v4_rmsnorm | 4.1681 | 48.30 | 0.0121 | 0.000000 |

现象：v4 correctness 通过，当前实测比 v3 快约 `1.7%`，比 v2 快约 `13.0%`。v4 和 v3 的有效访存字节相同，所以这点差异主要来自统计阶段更简单：v4 只做 `sum(x^2)` 的 `float` reduce，而 v3 要做 Welford update/combine。

### 代价或限制
RMSNorm 不做 mean centering，因此不能直接替换所有 LayerNorm 场景。它适合模型结构本来就使用 RMSNorm 的情况；若模型需要 LayerNorm 的零均值输出，v4 的数学语义就不等价。

本版也没有加入 `gamma` affine scale。真实 RMSNorm 通常会有逐列权重 `gamma[col]`，下一步若继续归一化算子，可以做 fused affine；如果目标是推进 CUDA 主线，现在也适合进入 GEMM。

### v4 作业
1. 概念题：v4 和 v3 都是两次读 `x`、一次写 `y`，为什么 v4 仍可能略快？请从统计量、归约对象和“算子语义不同”三个角度回答。
2. 改错题：下面这段 RMSNorm 代码至少有两个问题，请指出并说明后果。

```cpp
float thread_sq_sum = 0.0f;
for (int col = tid; col < cols; col += blockDim.x) {
    thread_sq_sum += x[row_offset + col];
}

const float sq_sum = block_reduce_sum(thread_sq_sum);
const float inv_rms = 1.0f / sqrtf(sq_sum + kEps);

for (int col = tid; col < cols; col += blockDim.x) {
    y[row_offset + col] = (x[row_offset + col] - mean) * inv_rms;
}
```

## 对比总表

注意：v1-v3 是 LayerNorm，v4 是 RMSNorm，数学定义不同。v4 的耗时不能解读为“LayerNorm v4 的纯优化收益”，只能用于比较两个归一化变体在当前实现下的统计开销。

| version | 核心手段 | 逻辑读写 | ms | GB/s | TFLOPS | max_err | 瓶颈 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程处理一行 | 读 `x` 3 次，写 `y` 1 次 | 99.5988 | 2.70 | 0.0010 | 0.000003 | latency-bound + low row-level parallelism |
| v2_block_warp | 一个 block 处理一行，warp shuffle block reduce | 读 `x` 3 次，写 `y` 1 次 | 4.7923 | 56.01 | 0.0210 | 0.000000 | memory-bound + reduction/sync overhead |
| v3_welford | Welford 一次统计 mean/variance | 读 `x` 2 次，写 `y` 1 次 | 4.2384 | 47.50 | 0.0277 | 0.000000 | memory-bound + Welford arithmetic/reduction overhead |
| v4_rmsnorm | RMSNorm，只统计 `sum(x^2)` | 读 `x` 2 次，写 `y` 1 次 | 4.1681 | 48.30 | 0.0121 | 0.000000 | memory-bound + simple reduction overhead |

## 参考资料
- CUDA C++ Programming Guide：thread hierarchy、global memory coalescing、shared memory。
- NVIDIA Nsight Compute CLI：后续 profiler 验证使用 `ncu` metric。
