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

## v2 — shared memory block reduce
### 改动
v2 让每个 block 先在 shared memory 中做树形归约，只由 `threadIdx.x == 0` 的线程把该 block 的局部和写回全局输出。

shared memory 是 block 内线程共享的片上存储。这里使用它的目的不是复用输入数据，而是把同一个 block 内的 `blockDim.x` 次全局原子加法，合并成 1 次全局原子加法。

### 代码要点
- 每个线程把 `x[idx]` 读入 `smem[tid]`；越界线程写 `0.0f`，保证尾部 block 正确。
- `__syncthreads()` 保证所有线程写完 shared memory 后，再开始归约。
- 每轮令 `stride` 减半，`tid < stride` 的线程执行 `smem[tid] += smem[tid + stride]`。
- 最后 `tid == 0` 执行一次 `atomicAdd(y, smem[0])`。

### 定量分析
设 `B = blockDim.x`，grid 数量约为 `ceil(N / B)`。

逻辑访存字节仍近似为：
- 输入读：`N * 4`
- 输出写：最终标量 `4`
- 逻辑字节 `B_bytes = N * 4 + 4`

FLOPs 仍是 `F = N - 1`，所以 `AI = F / B_bytes ≈ 0.25 FLOP/Byte`。AI 没有明显变化，因为算法最终还是每个输入元素参与一次加法。

真正变化的是全局原子次数：

| version | 全局 `atomicAdd` 次数 |
| --- | --- |
| naive | `N` |
| v2 | `ceil(N / blockDim.x)` |

默认 `N = 1 << 22`、`blockDim.x = 256` 时，原子次数从 `4,194,304` 次降到 `16,384` 次，约减少 256 倍。理论上这会显著缓解同地址 atomic contention。

但 v2 引入了新代价：每个 block 内有 `log2(blockDim.x)` 轮 `__syncthreads()`，并且归约后半段很多线程处于空闲状态。因此它通常会比 naive 快很多，但还不是最优 reduce。

瓶颈定性判定：从 **atomic contention-bound** 转向 **同步开销 + shared memory 归约开销 + 剩余全局 atomic contention** 混合限制。若 v2 的有效 `GB/s` 仍明显低于 elementwise，原因通常不是 DRAM 连续带宽不够，而是 block 内同步和最后的全局原子仍在限制吞吐。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

shared memory 读写和 `__syncthreads()` 的行级开销，可以优先在 Nsight Compute 的 Source 页面按代码行查看；这里不写未验证的 shared memory 专用 metric 名。

### 实测结果
本轮在当前环境中编译通过，但运行失败：

```text
CUDA driver version is insufficient for CUDA runtime version
```

因此 v2 的 `ms / GB/s / TFLOPS / max_err` 需要在 CUDA driver 与 runtime 匹配的机器上复测：

```bash
nvcc src/reduce.cu -o debugger/reduce && ./debugger/reduce
```

预期现象：`v2_smem` 应明显快于 `naive`。如果没有明显变快，优先检查 `max_err` 是否异常、`blockDim.x` 是否太小、以及 Nsight Compute Source 页面里 `atomicAdd` 和 `__syncthreads()` 的耗时占比。

### 瓶颈
v2 的主要瓶颈不再是每个元素都争抢全局 `y`，而是：
- block 内树形归约每一轮都需要同步；
- stride 变小时，活跃线程数逐轮减少；
- 每个 block 仍要对同一个全局地址做一次 `atomicAdd`。

这说明 v2 解决了最粗的全局原子竞争，但还没有解决“block 内归约效率”和“跨 block 最终合并”两个问题。

### 下一步
下一版可以让每个线程先读取并累加多个元素，再进入 shared memory 归约。这样能减少 block 数、减少最后的全局 `atomicAdd` 次数，并提高每个线程的工作量。

## 对比总表
| version | ms | GB/s | TFLOPS | 说明 |
| --- | --- | --- | --- | --- |
| naive | 8.3756 | 2.00 | 0.0005 | 每个线程一次全局 `atomicAdd`，正确但竞争严重 |
| v2_smem | 待实测 | 待实测 | 待实测 | block 内 shared memory 树形归约，每个 block 一次全局 `atomicAdd` |

## 作业
本算子的作业在 `homework/reduce.md`。

### v1 作业
1. 合上代码，默写 `kernel_naive` 的核心逻辑：算 `idx`、边界检查、用 `atomicAdd` 累加到输出标量。
2. 用一段话解释：为什么 reduce naive 版的 `AI` 比 vector add 高一点，但实际可能比 vector add 慢很多？

### v2 作业
1. 用一句话解释：v2 的 `AI` 为什么几乎没变，但性能预期会明显变好？
2. 找 bug：下面这段归约代码有什么问题？至少指出 1 个。

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] += smem[tid + stride];
    }
}
```

## 参考资料
- CUDA C++ Programming Guide: Atomic Functions
- Nsight Compute: Memory Workload Analysis / Source Counters
