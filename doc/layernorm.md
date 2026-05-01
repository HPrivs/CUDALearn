# LayerNorm / RMSNorm

## 学习目标
- 写出正确的行级 LayerNorm CUDA kernel。
- 理解 LayerNorm 的 mean、variance 和 normalize 三个阶段。
- 建立 naive multi-pass 的 correctness、benchmark 和定量分析基线。
- 明确哪些后续优化是复用旧技巧，哪些才是本算子的新学习重点。

## 前置知识
- row-major：二维数组按行连续存储，`x[row, col]` 的线性下标是 `row * cols + col`。
- mean：一组数的平均值，LayerNorm 中按每一行独立计算。
- variance：一组数围绕 mean 的平均平方偏差，反映这一行的离散程度。
- epsilon：加在 variance 上的小正数，用来避免除以 0 并改善数值稳定性。

## 问题规格
- 输入：矩阵 `x`，形状为 `M x K`
- 输出：矩阵 `y`，形状为 `M x K`
- dtype：`float32`
- 数学定义：`y[row, col] = (x[row, col] - mean(row)) / sqrt(var(row) + eps)`
- `mean(row) = sum_j x[row, j] / K`
- `var(row) = sum_j (x[row, j] - mean(row))^2 / K`
- 默认规模：`M = 4096, K = 4096`
- 默认 `eps = 1e-5`
- 存储布局：`x/y` 都是 row-major

本轮暂时不加入 `gamma/beta`，也不做 RMSNorm。首版目标是把 LayerNorm 本身的三阶段公式、CPU reference 和 benchmark 跑通。

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

当前记录，默认 `M = 4096, K = 4096`，`timeit` 使用 `warmup=3, iters=20`。本轮连续跑两次，下表取最小值：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 99.3326 | 2.70 | 0.0010 | 0.000003 |

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
下一版不重复单独讲 shared memory reduce 和 warp shuffle。我们会把 block-per-row、shared memory 和 `__shfl_down_sync` 合并成一个 optimized baseline，用 benchmark 说明这些旧技巧迁移到 LayerNorm 后的净效果。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、求 `mean`、求 `var`、计算 `inv_std`、写回 `y[row, col]`。
2. 用一段话解释：为什么 naive LayerNorm 的有效 `GB/s` 很低？请至少从“每行并行度”和“warp 内访存是否连续”两个角度回答。

## 对比总表

| version | 核心手段 | 逻辑读写 | ms | GB/s | TFLOPS | max_err | 瓶颈 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| naive | 一个线程处理一行 | 读 `x` 3 次，写 `y` 1 次 | 99.3326 | 2.70 | 0.0010 | 0.000003 | latency-bound + low row-level parallelism |

## 参考资料
- CUDA C++ Programming Guide：thread hierarchy、global memory coalescing、shared memory。
- NVIDIA Nsight Compute CLI：后续 profiler 验证使用 `ncu` metric。
