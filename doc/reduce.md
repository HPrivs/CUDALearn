# Reduce / Sum

## 问题规格
- 输入：一个长度为 `N` 的 1D 向量 `X`
- 输出：一个标量 `Y`
- dtype：`float32`
- 数学式：`Y = sum(X[i])`
- 默认规模：`N = 1 << 22`

## v1 — naive
### 改动
首版使用最直接的写法：一个线程读取一个元素，然后用 `atomicAdd` 把值加到同一个全局输出标量 `Y` 上。

`atomicAdd` 是原子加法：多个线程同时更新同一个地址时，硬件保证每次加法不会互相覆盖。

### 代码要点
- `idx = blockIdx.x * blockDim.x + threadIdx.x`
- 若 `idx < N`，执行 `atomicAdd(y, x[idx])`
- 每次 launch 前先把 `y` 清零，否则多轮 benchmark 会把结果累加到旧结果上
- `cpu_ref(...)` 用 `double` 做中间累加，再转回 `float`，减少 CPU 参考值自身的累加误差

### 定量分析
- 逻辑访存字节 `B = N * 4 + 4`，读 `N` 个 `float`，写 1 个输出标量
- FLOPs `F = N - 1`
- 算术强度 `AI = F / B ≈ 1 / 4 = 0.25 FLOP/Byte`

只看逻辑字节，reduce 的 AI 仍然很低，像 memory-bound。但 naive 版还有一个更严重的问题：所有线程都对同一个地址做 `atomicAdd`，这些更新会高度串行化。因此它通常不是单纯的 DRAM 带宽瓶颈，而是 **latency-bound / atomic contention-bound**：主要受原子操作延迟和同地址竞争限制。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

原子竞争本身可以结合 Nsight Compute 的 Source 页面看 `atomicAdd` 所在行的耗时占比；这里不写未验证的 atomic 专用 metric 名。

### 实测结果
先用下面命令在你的机器上测：

```bash
nvcc src/reduce.cu -o debugger/reduce && ./debugger/reduce
```

把输出表里的 `ms / GB/s / TFLOPS / max_err` 记下来。这里的 `GB/s` 是按逻辑读写字节估算的有效带宽，不等于原子操作实际造成的所有内存事务。

### 瓶颈
这个版本的瓶颈是全局原子操作竞争。虽然每个线程只做一次加法，但所有线程都争抢同一个 `y` 地址，硬件必须维护原子更新顺序，导致大量线程等待。

如果实测 `GB/s` 远低于 vector add，不代表显存带宽突然变差，而是因为 kernel 时间主要消耗在 `atomicAdd` 的串行化等待上。

### 下一步
下一版会使用 shared memory 做 block 内树形归约。目标是让每个 block 先把自己的局部和算出来，再减少全局 `atomicAdd` 的次数。

## 对比总表
| version | ms | GB/s | TFLOPS | 说明 |
| --- | --- | --- | --- | --- |
| naive | 待实测 | 待实测 | 待实测 | 每个线程一次全局 `atomicAdd`，正确但竞争严重 |

## 作业
本算子的作业在 `homework/reduce.md`。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：算 `idx`、边界检查、用 `atomicAdd` 累加到输出标量。
2. 用一段话解释：为什么 reduce naive 版的 `AI` 比 vector add 高一点，但实际可能比 vector add 慢很多？

## 参考资料
- CUDA C++ Programming Guide: Atomic Functions
- Nsight Compute: Memory Workload Analysis / Source Counters
