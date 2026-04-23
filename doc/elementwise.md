# ElementWise

## 问题规格
- 输入：两个长度为 `N` 的 1D 向量 `A`、`B`
- 输出：一个长度为 `N` 的 1D 向量 `C`
- dtype：`float32`
- 数学式：`C[i] = A[i] + B[i]`
- 默认规模：`N = 1 << 24`

## v1 — naive
### 改动
首版只做最直接的映射：一个线程处理一个元素，做一次加法后写回全局内存。

### 代码要点
- `idx = blockIdx.x * blockDim.x + threadIdx.x`
- 若 `idx < N`，执行 `c[idx] = a[idx] + b[idx]`
- `launch_naive(...)` 统一封装 kernel 启动，后面加新版本时可以直接放进 benchmark 循环

### 定量分析
- 访存字节 `B = N * (4 + 4 + 4) = 12N Bytes`
- FLOPs `F = N`
- 算术强度 `AI = F / B = 1 / 12 ≈ 0.083 FLOP/Byte`
- 这是非常低的 AI，通常会是 **memory-bound**：时间主要花在搬运 `A/B/C`，不是花在加法本身
- 可验证的 NCU metric：
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
  - `dram__bytes.sum`
  - `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

### 实测结果
先用下面命令在你的机器上测：

```bash
nvcc src/elementwise.cu -o elementwise && ./elementwise
```

把输出表里的 `ms / GB/s / max_err` 记下来，后面 v2、v3 都要和它比。

### 瓶颈
这个版本的瓶颈不是计算，而是 HBM / DRAM 带宽。每个元素只做 1 次加法，却要读 2 个 `float`、写 1 个 `float`。

### 下一步
下一版最自然的方向是 **向量化访存**：让每个线程一次搬 16B，例如 `float4`，减少指令条数并改善访存效率。

## v2 — `float4` 向量化访存
### 改动
这一版不再让一个线程只处理 1 个 `float`，而是让一个线程一次处理 4 个连续元素。具体做法是把 `float*` 按 `float4*` 解释，然后每个线程完成一次 `float4` 的 load、4 次加法、一次 `float4` 的 store。

这版没有改变总访存字节数，也没有改变总 FLOPs；它优化的是**访存指令组织方式**，不是算法本身。

### 代码要点
- 新增 `kernel_v2_float4(...)`，输入输出类型改为 `const float4* / float4*`
- 线程索引 `idx` 现在对应的是“第几个 `float4`”，不是“第几个标量元素”
- kernel 内先读 `av = a[idx]`、`bv = b[idx]`，再构造一个新的 `float4` 写回
- `launch_v2_float4(...)` 先处理 `n / 4` 个完整向量，再用 `kernel_naive` 处理尾部 `n % 4` 个元素

这里单独保留尾部标量路径，是因为真实代码不能默认 `N` 永远是 4 的倍数。这样写以后，这个版本既能跑默认规模，也能跑一般规模。

### 定量分析
- 访存字节仍然是 `B = 12N Bytes`
- FLOPs 仍然是 `F = N`
- 算术强度仍然是 `AI = 1 / 12 ≈ 0.083 FLOP/Byte`

这说明一个关键点：`float4` **不会改变这个算子的理论瓶颈类型**。它仍然是一个低 AI 的 elementwise kernel，主瓶颈仍应是 global memory 带宽，而不是计算吞吐。

那为什么它仍可能更快？因为 naive 版每处理 4 个元素，要做 4 次标量 load `A`、4 次标量 load `B`、4 次标量 store `C`；向量化后，同样 4 个元素可以收敛成 1 次 `float4` load、1 次 `float4` load、1 次 `float4` store。总字节没变，但访存指令条数更少，地址计算和发射开销也更低。

可验证的 NCU metric：
- `dram__bytes.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`

预期现象是：`dram__bytes.sum` 基本不变，`smsp__inst_executed.sum` 下降，而有效 `GB/s` 可能上升。如果 `GB/s` 几乎不变，说明原来的 naive 版已经比较接近带宽瓶颈，此时向量化能压缩的空间就有限。

### 实测结果
用下面命令重新编译并实测：

```bash
nvcc src/elementwise.cu -o debugger/elementwise && ./debugger/elementwise
```

重点比较两行：
- `naive` 的 `ms / GB/s`
- `v2_float4` 的 `ms / GB/s`

如果 `v2_float4` 更快，不要只记“向量化有效”，而要结合 profile 去确认：到底是 DRAM 吞吐更高了，还是只是指令数更少了。

### 瓶颈
这一版大概率仍是 **memory-bound**，只是更接近“把带宽吃满”的写法。换句话说，它更像是在减少访存相关的指令和调度开销，而不是把问题从 memory-bound 变成 compute-bound。

这一判断依赖两个前提：
- 数据地址满足 `float4` 对齐。这里 `cudaMalloc` 分配的地址通常满足这一点
- 数据规模足够大，能让 kernel 进入稳定吞吐区；太小的 `N` 下，launch latency 可能掩盖差异

### 下一步
如果继续做 `elementwise`，下一步可以考虑两个方向：
- **grid-stride loop**：让一个线程在大规模输入上处理多个 chunk，学习更通用的 kernel 写法
- **半精度或更复杂表达式融合**：例如 `C = alpha * A + beta * B`，观察 elementwise fuse 后 AI 和瓶颈会不会变化

## 对比总表
| version | ms | GB/s | TFLOPS | 说明 |
| --- | --- | --- | --- | --- |
| naive | 4.1703 | 48.28 | 0.0040 | 一个线程处理一个元素 |
| v2_float4 | 4.1346 | 48.68 | 0.0041 | 一个线程处理 4 个连续元素，减少访存指令条数 |

## 作业
本算子的作业已迁移到 `homework/elementwise.md`。

你可以在该文件里直接写：
- 代码块
- 概念题回答
- 自己的分析过程

批改时如果你说「批改 elementwise 作业」，默认读取 `homework/elementwise.md`。
