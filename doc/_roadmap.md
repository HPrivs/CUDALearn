# CUDA 算子学习路线

## ✅ 已学 / 进行中

### Elementwise / VecAdd（入门首选）
1. naive：一个线程处理一个元素
2. `float4` 向量化访存：一个线程处理 4 个连续元素，减少访存指令条数

**瓶颈演进**：始终 memory-bound，目标逼近 DRAM 带宽峰值

**后续参考**：可选做 scalar/unroll 对照、对齐与非对齐 `float4` 对照，用来理解向量化访存的适用条件。

### Reduce / Sum（入门第二步）
1. naive：atomicAdd 到全局变量（正确但极慢）
2. shared memory block reduce：每个 block 先做局部归约，再执行一次全局 atomicAdd
3. per-thread accumulation：每个线程先在 register 中累加多个元素，再进入 shared memory 归约
4. two-pass reduction：第一轮写 partial sums，第二轮归约 partial，去掉跨 block 同地址 atomic 竞争
5. warp shuffle block reduce：warp 内用 `__shfl_down_sync` 归约，减少 shared memory 读写和部分同步

**当前瓶颈**：输入读取 + 额外 kernel launch 开销 + 少量 block 内归约开销

**后续参考**：可选做 vectorized load + multiple elements per thread，对比是否真的受 global load 指令数量限制。

### Transpose（二维访存入门）
1. naive：一个线程搬运一个元素，读连续、写跨 stride
2. shared memory tiled transpose：用 shared memory 暂存 tile，交换 block 坐标后连续写回
3. shared memory padding：给 tile 行跨度加 1，减少转置读取时的 bank conflict

**当前瓶颈**：仍是 memory-bound；padding 在当前 GP108 上未稳定提速，下一步更适合转向 GEMV

**后续参考**：可选做 vectorized transpose 或不同 tile shape，对比 global transaction 与 occupancy 的取舍。

### GEMV（矩阵×向量）
1. naive：一个线程计算一整行 dot product
2. block-per-row reduce：一个 block 负责一行，多个线程并行计算 partial sum，再做 block 内归约
3. multi-row per block：一个 block 同时处理多行，减少 block 数并观察每行线程数的取舍

**当前瓶颈**：memory-bound + block 粒度取舍；`x` 的实际 DRAM 复用还需要 profiler 验证。

**后续参考**：不同 `kRowsPerBlock` 参数实验、vectorized load + unroll、cache / read-only path 观察，并用 profiler 判断是否真的受 global load 指令和 DRAM traffic 限制。

### Softmax（行归约 + 指数归一化）
1. naive multi-pass：一个线程处理一行，分别求 max、求 exp sum、写归一化结果
2. block-per-row shared memory reduce：一个 block 处理一行，用 shared memory 做 max 和 sum 归约
3. warp shuffle block reduce：warp 内用 `__shfl_down_sync` 归约，减少 shared memory 读写和部分同步
4. online softmax：把 max 和 sum 的统计合并到一次扫描，减少一次 global memory 读取

**当前瓶颈**：latency-bound + online rescale/reduce overhead；global memory 扫描已从三次降到两次，但写回阶段仍要再次读取 `x` 并计算 `expf`。

**后续参考**：vectorized load/store，或进入 Attention，把 online softmax 用到分块 `QK^T` 上。

### LayerNorm / RMSNorm（归一化层）
1. naive multi-pass LayerNorm：一个线程处理一行，分别求 mean、variance，再归一化写回

**当前瓶颈**：latency-bound + low row-level parallelism + uncoalesced global access；有效 GB/s 很低，说明 naive 版没有打满 DRAM 带宽。

**后续参考**：下一步把已学过的 block-per-row、shared memory 和 `__shfl_down_sync` 合并成 optimized baseline；后续重点放在 Welford variance、fused affine 和 RMSNorm。

---

## 🔜 下一步候选

下面步骤是路线参考，不代表已经实现或已经实测；实际学习时仍然每轮只引入一个主要优化手段。后续算子可以把已经学过的优化手段作为 baseline 组合使用；每个步骤只标出本轮真正新增的核心概念。

### GEMM（矩阵乘法）
学习价值：串联二维 tiling、shared memory 复用、register tile 和更高阶的矩阵乘优化。

推荐步骤：
1. naive：一个线程计算一个 `C[row, col]`
2. shared memory tile：把 `A/B` tile 搬到 shared memory，减少 global memory 重复读取
3. register tile：一个线程计算多个输出元素，提高每次读入数据的复用
4. tile 参数与 occupancy / register pressure 分析：解释 tile 过大为什么可能变慢
5. vectorized global load：用连续向量化访存改善 global load/store 指令效率
6. double buffering：计算当前 tile 时预取下一 tile，隐藏部分访存延迟
7. `cp.async`（硬件支持时）：异步拷贝 global 到 shared memory，减少同步等待
8. Tensor Core（硬件支持时）：用 WMMA / MMA 提高低精度矩阵乘吞吐

### Conv2d（二维卷积）
学习价值：理解卷积如何转化或映射成矩阵乘，并观察数据复用与边界处理。

推荐步骤：
1. naive direct conv：一个线程计算一个输出元素，先保证 padding/stride/dilation 正确
2. shared memory input tile：缓存输入 patch，减少相邻输出重复读取
3. constant / read-only filter：小卷积核权重走更适合广播或缓存的路径
4. im2col + GEMM：把卷积转成矩阵乘，复用 GEMM 优化经验
5. implicit GEMM：不显式落地 im2col，减少额外 HBM 读写
6. Winograd（小 kernel 可选）：用更多变换换取更少乘法，重点分析适用范围

### Attention（Q/K/V 分块 + online softmax）
学习价值：把 GEMM、Softmax 和分块访存融合起来，理解为什么 FlashAttention 能减少 HBM 往返。

推荐步骤：
1. naive attention：显式计算 `S = QK^T`、softmax、再乘 `V`
2. tiled QK：分块计算 score，建立 Q/K tile 的 shared memory 复用
3. online softmax per block：分块更新每行 max 和 sum，避免完整 `S` 落 HBM
4. fused `P * V` accumulation：softmax 结果不完整写回，直接累加输出
5. causal mask / padding mask：加入真实模型常见边界条件
6. double buffering / Tensor Core（硬件支持时）：进一步提高 tile 计算吞吐
