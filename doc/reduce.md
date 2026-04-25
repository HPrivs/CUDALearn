# Reduce / Sum

## 学习目标
- 写出正确的 sum reduce CUDA kernel。
- 理解全局 `atomicAdd` 为什么能保证正确但会形成严重竞争。
- 用 shared memory 做 block 内局部归约，减少全局原子操作次数。
- 区分 `AI` 几乎不变和实际瓶颈明显变化这两件事。
- 用 two-pass reduction 去掉跨 block 合并阶段的同地址全局原子竞争。
- 用 warp-level reduction 减少 block 内 shared memory 访问和部分 `__syncthreads()`。

## 前置知识
- reduce：把多个输入元素合并成更少输出元素的操作。
- `atomicAdd`：多个线程更新同一个地址时，硬件保证每次读-改-写不会互相覆盖。
- shared memory：同一个 block 内线程共享的片上存储。
- register：每个线程私有的最快片上存储，适合保存该线程自己的临时累加值。

## 问题规格
- 输入：长度为 `N` 的 1D 向量 `X`
- 输出：1 个标量 `Y`
- dtype：`float32`
- 数学定义：`Y = sum(X[i])`
- 默认规模：`N = 1 << 24`

Reduce 和 elementwise 的关键区别是：输出元素数量远少于输入元素数量。很多线程必须把结果合并到同一个标量上，因此性能问题通常不只是“读得够不够快”，还包括“怎么合并得够快”。

## v1 — naive

### 本版学习目标
先写出最直接的正确 reduce：每个线程读一个元素，然后用全局 `atomicAdd` 合并。

### 改动
首版使用最直接的正确写法：一个线程读取一个输入元素，然后用 `atomicAdd` 把它累加到全局输出标量 `Y`。

`atomicAdd` 是原子加法：多个线程同时更新同一个地址时，硬件保证每次读-改-写不会互相覆盖。

### 为什么可能更快
这是 correctness 基线，不追求性能；它用最少控制逻辑换取正确的跨线程合并。

### 代码要点
- `idx = blockIdx.x * blockDim.x + threadIdx.x` 得到当前线程负责的元素。
- `idx < N` 时执行 `atomicAdd(y, x[idx])`。
- 每次 launch 前必须 `cudaMemset(y, 0, sizeof(float))`，否则 benchmark 的多轮运行会把结果累加到旧值上。
- `cpu_ref(...)` 用 `double` 做中间累加，再转回 `float`，降低 CPU 参考值自身的舍入误差。

### 定量分析
只看算法的逻辑读写：

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `N * 4 + 4` | 读 `N` 个 `float`，最终得到 1 个 `float` 输出 |
| FLOPs `F` | `N - 1` | 求和需要 `N - 1` 次加法 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 仍然很低 |

按 `AI` 看，reduce 像 memory-bound：每读 4 字节大约只做 1 次加法。但 naive 版更主要的问题不是普通 DRAM 带宽，而是所有线程都对同一个地址做 `atomicAdd`。这些原子更新必须维护顺序，实际会形成严重的同地址竞争。

瓶颈定性判定：**latency-bound / atomic contention-bound**。主要限制来自全局原子操作的排队、重试和等待，而不是连续读 `x` 的带宽。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

原子竞争可以先在 Nsight Compute 的 Source 页面看 `atomicAdd` 所在行的耗时占比。这里不写未验证的 atomic 专用 metric 名。

### 实测结果
当前记录：

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 31.5855 | 2.12 | 0.0005 | 0.076721 |

这里的 `GB/s` 是按逻辑字节 `N * 4 + 4` 计算的有效带宽，不等于原子操作真实产生的所有内存事务。它很低时，不能直接解释为“显存带宽差”，更合理的解释是 kernel 时间主要花在全局原子竞争上。

### 瓶颈
naive 版的输入读取是连续的，相邻线程访问相邻 `x[idx]`，这部分访存模式并不差。真正拖慢的是输出侧：所有线程都争抢同一个 `y` 地址。

因此这个版本适合作为正确性基线，但不适合作为性能基线。它告诉我们第一件要优化的事不是减少输入读，而是减少全局原子加法次数。

### 代价或限制
全局 `atomicAdd` 次数随 `N` 线性增长，输入越大，同地址竞争越严重。

### 下一步
下一版使用 shared memory 做 block 内树形归约。目标是让每个 block 先算出自己的局部和，再把 `N` 次全局 `atomicAdd` 降为约 `ceil(N / blockDim.x)` 次。

## v2 — shared memory block reduce

### 本版学习目标
用 shared memory 在 block 内先合并局部结果，把全局 `atomicAdd` 从每元素一次降到每 block 一次。

### 改动
v2 让每个 block 先在 shared memory 中做树形归约，最后只由 `threadIdx.x == 0` 的线程把该 block 的局部和加到全局输出。

shared memory 是同一个 block 内线程共享的片上存储。这里用它不是为了复用输入 `x`，而是为了把 block 内许多次全局原子加法合并成 1 次。

### 代码要点
- 每个线程先把 `x[idx]` 写入 `smem[tid]`；越界线程写 `0.0f`，保证尾部 block 正确。
- 第一个 `__syncthreads()` 保证所有线程完成写入后，才开始读别人的 `smem`。
- 每轮 `stride` 减半，`tid < stride` 的线程执行 `smem[tid] += smem[tid + stride]`。
- 每轮归约后都需要 `__syncthreads()`，否则下一轮可能读到尚未写完的数据。
- 最后只有 `tid == 0` 执行 `atomicAdd(y, smem[0])`。

### 定量分析
设 `T = blockDim.x`，grid 大小为 `ceil(N / T)`。

逻辑读写和 FLOPs 基本没变：

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `N * 4 + 4` | 仍然主要是读输入 |
| FLOPs `F` | `N - 1` | 每个元素仍参与一次求和 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 与 naive 几乎相同 |

v2 的核心收益不体现在 `AI`，而体现在全局原子次数：

| version | 全局 `atomicAdd` 次数 |
| --- | ---: |
| naive | `N` |
| v2_smem | `ceil(N / T)` |

默认 `N = 1 << 24`、`T = 256` 时，原子次数从 `16,777,216` 次降到 `65,536` 次，约减少 256 倍。这会显著缓解同地址 atomic contention。

代价也很明确：block 内树形归约需要 `log2(T)` 轮同步；stride 变小时，参与加法的线程数每轮减半，后半段有大量线程空闲；每个 block 最后仍要对同一个全局 `y` 做一次 `atomicAdd`。

瓶颈定性判定：**同步开销 + shared memory 归约开销 + 剩余全局 atomic contention**。它通常会比 naive 快很多，但还不是理想 reduce。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

shared memory 访问和 `__syncthreads()` 的行级影响，优先在 Nsight Compute 的 Source 页面按代码行看耗时。不要只看 `AI`，因为 v2 的优化目标不是提高 `AI`，而是减少全局合并冲突。

### 实测结果

```bash
nvcc src/reduce.cu -o debugger/reduce && ./debugger/reduce
```

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| v2_smem | 6.1032 | 11.00 | 0.0027 | 0.004395 |

预期现象：`v2_smem` 应明显快于 `naive`。如果没有明显变快，按下面顺序检查：

1. `max_err` 是否异常，先排除正确性问题。
2. `blockDim.x` 是否过小，过小会导致 block 数过多，剩余全局原子次数仍然多。
3. Source 页面中 `atomicAdd` 和 `__syncthreads()` 所在行是否占主要时间。

如果实测与预期不一致，记录格式建议写成：

```text
现象：v2_smem 没有明显快于 naive。
可能原因：全局 atomicAdd 仍占较高比例，或 block 内同步开销抵消了减少 atomic 的收益。
下一步验证：看 Source 页面中 atomicAdd 与 __syncthreads 的行级耗时，并尝试调整 blockDim.x。
```

### 瓶颈
v2 已经解决最粗糙的全局原子风暴，但仍有三个限制：

- 每个 block 的归约需要多轮 `__syncthreads()`。
- 树形归约后半段线程利用率逐轮下降。
- 跨 block 合并仍然依赖对同一个全局地址的 `atomicAdd`。

这说明下一步应该同时考虑两个方向：减少参与 block 内归约的数据量，或者减少最后写全局输出的竞争。

### 下一步
下一版可以让每个线程先读取并累加多个元素，再进入 shared memory 归约。这样做会减少 block 数、减少最后的全局 `atomicAdd` 次数，并提高每个线程在同步前完成的有效工作量。

## v3 — per-thread accumulation

### 本版学习目标
让每个线程先在 register 中累加多个元素，再把局部和写入 shared memory 做 block 内归约。

register 是每个线程私有的最快片上存储。这里用它保存该线程的临时 `sum`，避免每读一个元素都立刻进入 block 级合并。

### 改动
v2 中每个线程只读 1 个元素，每个 block 处理 `blockDim.x` 个元素。v3 中每个线程读最多 `kItemsPerThread = 16` 个元素，并在自己的 `sum` 变量里先累加，因此每个 block 处理 `blockDim.x * 16` 个元素。

读取方式是：

```cpp
idx = block_start + i * blockDim.x + tid
```

对固定的 `i`，相邻线程仍然读取相邻地址，所以每一轮读都是 coalesced access。

### 为什么可能更快
v3 的收益来自两个地方：

1. block 数减少为 v2 的约 `1 / 16`。
2. 最终全局 `atomicAdd` 次数也减少为 v2 的约 `1 / 16`。

默认 `N = 1 << 24`、`blockDim.x = 256`、`kItemsPerThread = 16` 时：

| version | 每 block 处理元素 | 全局 `atomicAdd` 次数 |
| --- | ---: | ---: |
| v2_smem | 256 | 65,536 |
| v3_items16 | 4,096 | 4,096 |

这版没有减少读取 `x` 的 DRAM 字节数，也没有改变求和的总 FLOPs。它减少的是 block 级归约和跨 block 合并的次数。

### 代码要点
- `sum` 是线程私有 register 变量，先累加最多 16 个输入元素。
- 每个线程只把一个局部和写入 `smem[tid]`。
- shared memory 树形归约逻辑和 v2 相同。
- 尾部元素用 `idx < n` 保护，所以不要求 `N` 能被 `blockDim.x * kItemsPerThread` 整除。

### 定量分析
设 `T = blockDim.x`，`K = kItemsPerThread`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `N * 4 + 4` | 仍然读 `N` 个输入，最终得到 1 个输出 |
| FLOPs `F` | `N - 1` | 总求和次数不变 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 与 v2 基本相同 |

全局 `atomicAdd` 次数从 `ceil(N / T)` 降为 `ceil(N / (T * K))`。默认配置下是从 `65,536` 次降到 `4,096` 次。

瓶颈定性判定：**memory-bound + 剩余 block 归约开销 + 剩余全局 atomic contention**。如果 `K` 过大，单线程串行累加太多元素、寄存器压力增加，可能让 occupancy 或并行度下降。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

如果要验证这版是否减少了全局合并开销，优先比较 Source 页面中最后 `atomicAdd` 所在行的耗时占比，再结合总 kernel 时间判断。

### 实测结果
```bash
nvcc src/reduce.cu -o debugger/reduce && ./debugger/reduce
```

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| v3_items16 | 1.5780 | 42.53 | 0.0106 | 0.000244 |

实测现象：`v3_items16` 快于 `v2_smem`，但提升幅度没有达到 16 倍，因为 shared memory 归约、同步、输入读取和 kernel 调度仍然存在。


### 瓶颈
v3 进一步减少了全局 `atomicAdd` 和 block 数，但 block 内仍然使用 shared memory 树形归约，每轮仍有 `__syncthreads()`。当数据规模很大时，输入 DRAM 读取会逐渐成为更主要的限制。

### 代价或限制
`kItemsPerThread` 不是越大越好。它变大后，单线程串行工作增加，寄存器使用可能增加，block 数也会减少；如果 block 数过少，GPU 可能吃不满。

### 下一步
下一版先处理剩余全局 `atomicAdd` 竞争：用 two-pass reduction 把每个 block 的局部和写入 `partial` 数组，再用第二个 kernel 归约 `partial`。

## v4 — two-pass reduction

### 本版学习目标
把跨 block 合并从“多个 block 对同一个地址执行 `atomicAdd`”改成“两轮 kernel 明确归约”，理解用一次额外 kernel launch 换掉全局原子竞争的取舍。

two-pass reduction：第一轮产生多个局部结果，第二轮再把这些局部结果归约成最终结果。

### 改动
v4 保留 v3 的 per-thread accumulation 和 shared memory block reduce，但不再在第一轮末尾执行 `atomicAdd(y, smem[0])`。

第一轮 `kernel_v4_stage1`：
- 每个 block 处理 `blockDim.x * kItemsPerThread` 个输入。
- 每个 block 只把自己的局部和写到 `partial[blockIdx.x]`。

第二轮 `kernel_v4_stage2`：
- 用一个 block 读取所有 `partial`。
- block 内再次归约，最后由 `tid == 0` 写 `y[0]`。

### 为什么可能更快
v3 还有 `ceil(N / (blockDim.x * kItemsPerThread))` 次全局 `atomicAdd`，默认是 `4,096` 次，而且这些原子加法都争抢同一个 `y` 地址。

v4 第一轮每个 block 写不同的 `partial[blockIdx.x]`，没有同地址竞争；第二轮只用一个 block 归约 4,096 个 partial，最后普通写一次 `y[0]`。因此它主要去掉了跨 block 合并阶段的 atomic contention。

### 代码要点
- `d_partial` 在 `main()` 中分配，大小为 `partial_count = ceil(N / (blockDim.x * kItemsPerThread))`。
- `kernel_v4_stage1` 的读取和 block 内归约逻辑与 v3 基本相同，但最后写 `partial[blockIdx.x]`。
- `kernel_v4_stage2` 中每个线程用 `idx += blockDim.x` 的循环读取多个 partial，保证 `partial_count` 大于 `blockDim.x` 时也正确。
- `launch_v4` 连续启动两个 kernel：stage1 产生 partial，stage2 产生最终输出。

### 定量分析
设 `T = blockDim.x`，`K = kItemsPerThread`，`P = ceil(N / (T * K))`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `N * 4 + P * 4 + P * 4 + 4` | 读 `N` 个输入，写 `P` 个 partial，读 `P` 个 partial，写 1 个输出 |
| FLOPs `F` | `N - 1` | 仍然是求和所需的有效加法数 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 比 v3 略低一点，因为多了 partial 的读写 |

默认 `N = 1 << 24`、`T = 256`、`K = 16` 时，`P = 4,096`。额外 partial 读写约 `4,096 * 4 * 2 = 32 KiB`，相对输入的 `64 MiB` 很小。

瓶颈定性判定：**memory-bound + block 内同步开销 + 额外 kernel launch 开销**。v4 去掉了跨 block 同地址 atomic contention，但没有减少输入读取，也没有减少 block 内 shared memory 归约和 `__syncthreads()`。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

如果要验证 v4 的收益，优先比较 v3 和 v4 的总 kernel 时间，并在 Source 页面确认 v3 末尾 `atomicAdd` 不再出现在 v4 stage1 的热路径中。

### 实测结果

```bash
nvcc src/reduce.cu -o debugger/reduce && ./debugger/reduce
```

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| v4_two_pass | 1.3778 | 48.73 | 0.0122 | 0.000732 |

对比当前 `v3_items16 = 1.5780 ms`，v4 约快 `1.15x`。这里的 `GB/s` 对 v4 计入了 `partial` 的一次写和一次读，因此和 v3 的有效字节口径略有差异，但额外字节很小，不影响主要结论。默认 benchmark 现在使用 `warmup=3, iters=20`，因此小幅优化结论需要复测确认。

### 瓶颈
v4 已经把跨 block 合并中的全局 `atomicAdd` 去掉。剩下的主要开销是输入读取、每个 block 的 shared memory 树形归约、每轮 `__syncthreads()`，以及 two-pass 带来的第二次 kernel launch。

### 代价或限制
two-pass reduction 需要额外 `partial` 缓冲区和第二个 kernel。对于很小的 `N`，第二次 kernel launch 的固定开销可能大于去掉 atomic 的收益；对于很大的 `N`，这种代价通常可以被摊薄。

当前 `kernel_v4_stage2` 假设 `partial_count` 可以由一个 block 通过循环处理完。默认规模下没问题；如果 `N` 大到 `partial_count` 非常大，下一步需要做多级 reduction，而不是只用一个 stage2 block。

### 下一步
下一版可以引入 warp-level reduction，用 `__shfl_down_sync` 减少 shared memory 访问和部分 `__syncthreads()`，继续优化 block 内归约开销。

## v5 — warp shuffle block reduce

### 本版学习目标
用 `__shfl_down_sync` 做 warp-level reduction，理解 warp 内线程可以直接交换 register 值，从而减少 shared memory 读写和部分 block 级同步。

warp-level reduction：先在每个 warp 内完成局部归约，再把每个 warp 的结果交给一个 warp 做最后合并。

### 改动
v5 保留 v4 的 two-pass 结构：第一轮写 `partial`，第二轮归约 `partial`。本轮只替换 block 内归约方式。

v4 的 block reduce 做法是：所有线程把局部和写入 `smem[tid]`，然后执行 `128, 64, 32, ... 1` 的 shared memory 树形归约，每一轮都需要 `__syncthreads()`。

v5 的 block reduce 做法是：
- 每个 warp 内用 `__shfl_down_sync` 直接在 register 间交换并累加。
- 每个 warp 只由 lane 0 写一个 `warp_sums[warp_id]` 到 shared memory。
- 第一个 warp 再归约这些 warp partial sums。

### 为什么可能更快
默认 `blockDim.x = 256`，一个 block 有 8 个 warp。v4 每个 block 会把 256 个线程局部和都写进 shared memory，并经过 8 轮同步式树形归约。

v5 每个 block 只把 8 个 warp partial sums 写入 shared memory，中间只需要一次 `__syncthreads()` 让第一个 warp 看到这些结果。它减少的是片上 shared memory 读写和同步次数，不减少输入 DRAM 读取。

### 代码要点
- `warp_reduce_sum(value)` 用 offset `16, 8, 4, 2, 1` 做 5 轮 warp 内归约。
- `block_reduce_sum_v5(value, warp_sums)` 先做 warp 内归约，再用 shared memory 暂存每个 warp 的结果。
- `kernel_v5_stage1` 和 `kernel_v5_stage2` 的输入遍历逻辑与 v4 保持一致，只替换 block reduce helper。
- `kBlockSize = 256` 是 `warpSize = 32` 的整数倍，因此 `kWarpsPerBlock = 8`。

### 定量分析
设 `T = blockDim.x`，`K = kItemsPerThread`，`P = ceil(N / (T * K))`。

| 量 | 表达式 | 说明 |
| --- | --- | --- |
| 访存字节 `B` | `N * 4 + P * 4 + P * 4 + 4` | 与 v4 相同：读输入、写 partial、读 partial、写输出 |
| FLOPs `F` | `N - 1` | 有效求和次数不变 |
| 算术强度 `AI` | `F / B ≈ 0.25 FLOP/Byte` | 与 v4 基本相同 |

v5 的优化不体现在 `AI`，而体现在 block 内归约的执行方式：

| 项目 | v4 shared memory tree | v5 warp shuffle |
| --- | ---: | ---: |
| 每 block 写入 shared memory 的局部和数量 | 256 | 8 |
| block reduce 中主要同步次数 | 8 次左右 | 1 次 |
| 跨 block 合并 | two-pass，无全局 atomic | two-pass，无全局 atomic |

瓶颈定性判定：**memory-bound + kernel launch 开销 + 少量 block 内归约开销**。如果输入读取已经占主导，v5 可能只比 v4 小幅变快；如果 block 内归约和同步占比高，v5 的收益会更明显。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

`__shfl_down_sync` 的具体行级影响，优先在 Nsight Compute 的 Source 页面比较 v4 归约循环和 v5 helper 的耗时。这里不写未验证的 shuffle 专用 metric 名。

### 实测结果

```bash
nvcc src/reduce.cu -o debugger/reduce && ./debugger/reduce
```

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| v5_warp_shuffle | 1.3732 | 48.89 | 0.0122 | 0.000183 |

对比当前 `v4_two_pass = 1.3778 ms`，v5 只快约 `1.003x`。这是一个很小的差异，不能夸大成稳定大幅提升；需要提高 `iters` 或重复多轮才能确认小幅差距。

现象：v5 理论上减少了 shared memory 和同步，但实测只略快。

可能原因：当前 `N = 1 << 24` 时，输入读取和 two-pass 的 kernel launch 已经占较大比例；block 内归约只剩一部分开销，所以减少它不会带来大幅提升。

下一步验证：用 Nsight Compute Source 页面比较 v4 归约循环和 v5 `block_reduce_sum_v5` 的行级耗时；同时尝试较小 `N` 或不同 `kItemsPerThread`，观察 block reduce 开销占比变化。

### 瓶颈
v5 之后，Reduce 的主线优化已经从“消除全局 atomic 竞争”推进到“压缩 block 内归约开销”。当前主要剩余瓶颈更接近输入读取、kernel launch，以及浮点归约顺序导致的误差差异。

### 代价或限制
`__shfl_down_sync` 只在一个 warp 内交换数据，不能直接跨 warp 归约，所以仍然需要 shared memory 存 8 个 warp partial sums。代码也默认 block size 是 warp size 的整数倍；如果以后改成非 32 整数倍的 block size，需要重新检查 mask 和尾部线程。

### 下一步
Reduce 可以先到这里收束。下一个算子建议进入 Transpose，用矩阵转置学习 2D indexing、coalesced read/write，以及 shared memory bank conflict。

### v5 作业

1. 用一句话解释：v5 的 `AI` 为什么和 v4 基本一样，但仍然可能略快？
2. 找 bug：如果把 `block_reduce_sum_v5` 里 `__syncthreads()` 删除，结果为什么可能不正确？

## 对比总表

| version | 核心方法 | 全局 atomic 次数 | 主要瓶颈 | 当前状态 |
| --- | --- | ---: | --- | --- |
| naive | 每个线程直接 `atomicAdd(y, x[idx])` | `N` | 同地址全局原子竞争 | 已测：`31.5855 ms`，`2.12 GB/s` |
| v2_smem | block 内 shared memory 树形归约，每个 block 一次全局 `atomicAdd` | `ceil(N / blockDim.x)` | block 内同步 + 剩余全局原子竞争 | 已测：`6.1032 ms`，`11.00 GB/s` |
| v3_items16 | 每线程 register 预累加 16 个元素，再做 block reduce | `ceil(N / (blockDim.x * 16))` | 输入读取 + block 内同步 + 剩余全局原子竞争 | 已测：`1.5780 ms`，`42.53 GB/s` |
| v4_two_pass | 每个 block 写 partial，第二轮归约 partial | 0 | 输入读取 + block 内同步 + kernel launch | 已测：`1.3778 ms`，`48.73 GB/s` |
| v5_warp_shuffle | two-pass + warp shuffle block reduce | 0 | 输入读取 + kernel launch + 少量 block reduce | 已测：`1.3732 ms`，`48.89 GB/s` |

## 参考资料
- CUDA C++ Programming Guide: Atomic Functions
- CUDA C++ Programming Guide: Shared Memory
- Nsight Compute: Memory Workload Analysis / Source Counters
