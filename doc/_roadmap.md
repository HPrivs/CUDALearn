# CUDA 算子学习路线

说明：`reference/` 中的实现只作为后续路线和代码风格参考，不代表本项目已实现或已实测；若参考实现依赖 PyTorch extension、CuTe/CUTLASS、Triton 或特定架构指令，进入 Build Mode 时仍要改写成 `src/` 下可独立编译的教学版本。

## ✅ 已学 / 进行中

### Elementwise / VecAdd（入门首选）
1. naive：一个线程处理一个元素
2. `float4` 向量化访存：一个线程处理 4 个连续元素，减少访存指令条数

**瓶颈演进**：始终 memory-bound，目标逼近 DRAM 带宽峰值

**后续参考**：可选做 scalar/unroll 对照、对齐与非对齐 `float4` 对照，用来理解向量化访存的适用条件。若转向激活函数，可参考 `reference/kernels/relu/`、`sigmoid/`、`gelu/`、`swish/`：旧的 `float4` 技巧作为 baseline，新增重点放在 FP16/`half2`、128-bit packed load/store、exp/tanh 近似与数值范围裁剪。

### Reduce / Sum（入门第二步）
1. naive：atomicAdd 到全局变量（正确但极慢）
2. shared memory block reduce：每个 block 先做局部归约，再执行一次全局 atomicAdd
3. per-thread accumulation：每个线程先在 register 中累加多个元素，再进入 shared memory 归约
4. two-pass reduction：第一轮写 partial sums，第二轮归约 partial，去掉跨 block 同地址 atomic 竞争
5. warp shuffle block reduce：warp 内用 `__shfl_down_sync` 归约，减少 shared memory 读写和部分同步

**当前瓶颈**：输入读取 + 额外 kernel launch 开销 + 少量 block 内归约开销

**后续参考**：可参考 `reference/kernels/reduce/block_all_reduce.cu`。下一步不再重复完整 reduce 主线，优先做 `float4`/128-bit packed load、FP16 input + FP32 accumulation，或 INT8 packed reduction，用来观察 dtype、pack 宽度和累加精度的边界。

### Transpose（二维访存入门）
1. naive：一个线程搬运一个元素，读连续、写跨 stride
2. shared memory tiled transpose：用 shared memory 暂存 tile，交换 block 坐标后连续写回
3. shared memory padding：给 tile 行跨度加 1，减少转置读取时的 bank conflict

**当前瓶颈**：仍是 memory-bound；padding 在当前 GP108 上未稳定提速，下一步更适合转向 GEMV

**后续参考**：可参考 `reference/kernels/mat-transpose/` 和 `reference/kernels/swizzle/mat_trans_swizzle.cu`。推荐只保留一个进阶轮次：vectorized transpose + shared memory swizzle，对比 padding 与 swizzle 对 bank conflict 的适用边界。

### GEMV（矩阵×向量）
1. naive：一个线程计算一整行 dot product
2. block-per-row reduce：一个 block 负责一行，多个线程并行计算 partial sum，再做 block 内归约
3. multi-row per block：一个 block 同时处理多行，减少 block 数并观察每行线程数的取舍

**当前瓶颈**：memory-bound + block 粒度取舍；`x` 的实际 DRAM 复用还需要 profiler 验证。

**后续参考**：可参考 `reference/kernels/sgemv/` 和 `reference/kernels/hgemv/`。后续建议按 `K` 形态拆：warp-per-row for `K≈32`、`float4` for `K≈128`、multi-row per warp for small `K`，再选做 FP16/`half2` 路径；重点验证不同 `K` 下线程组织是否比 block-per-row 更合适。

### Softmax（行归约 + 指数归一化）
1. naive multi-pass：一个线程处理一行，分别求 max、求 exp sum、写归一化结果
2. block-per-row shared memory reduce：一个 block 处理一行，用 shared memory 做 max 和 sum 归约
3. warp shuffle block reduce：warp 内用 `__shfl_down_sync` 归约，减少 shared memory 读写和部分同步
4. online softmax：把 max 和 sum 的统计合并到一次扫描，减少一次 global memory 读取

**当前瓶颈**：latency-bound + online rescale/reduce overhead；global memory 扫描已从三次降到两次，但写回阶段仍要再次读取 `x` 并计算 `expf`。

**后续参考**：可参考 `reference/kernels/softmax/`。后续建议做一个 packed 版本：`float4` online safe softmax，或 FP16 input + FP32 accumulation + packed store；若继续到 Attention，则把 online state 的合并逻辑迁移到分块 `QK^T`。

### LayerNorm / RMSNorm（归一化层）
1. naive multi-pass LayerNorm：一个线程处理一行，分别求 mean、variance，再归一化写回
2. optimized baseline：一个 block 处理一行，复用 warp shuffle + shared memory 做行内归约
3. Welford variance：一次统计扫描同时得到 mean 和 variance，减少一次读取 `x`
4. RMSNorm：去掉 mean 统计，只归约 `sum(x^2)` 并按均方根缩放

**当前瓶颈**：memory-bound + row-level reduction overhead；v4 和 v3 一样是两次读 `x`、一次写 `y`，但 RMSNorm 的统计量更简单，当前默认规模下只比 v3 略快。

**后续参考**：可参考 `reference/kernels/layer-norm/` 和 `reference/kernels/rms-norm/`。下一步若继续归一化，优先做 fused affine（`gamma/beta` 或 RMSNorm `gamma`）+ `float4`/packed load；再考虑 FP16 input、FP32 accumulation、FP16 output 的混合精度版本。

### GEMM（矩阵乘法）
1. naive：一个线程计算一个 `C[row, col]`
2. shared memory tile：一个 block 负责一个输出 tile，把可复用的 `A/B` 片段缓存到 shared memory
3. register tile：一个线程计算同列两个输出元素，在寄存器中维护两个累加器并复用 shared memory 读出的 `B`
4. `2 x 2` register tile：一个线程计算 4 个输出元素，同时扩大输出 tile 的行和列，观察有效访存下降与 register/shared-memory pressure 的取舍
5. 收官版 `4 x 4` register tile：当前可验证的 FP32 手写 CUDA 版本；继续降低有效 DRAM 访存，同时记录 register pressure 和 occupancy 风险
6. cuBLAS SGEMM 对比基线：用库函数给手写 kernel 提供同机同规模的性能参考，不作为手写优化步骤

**当前瓶颈**：v5 是当前可实测 FP32 标量手写收官版；`cublas_sgemm` 作为库函数参考明显更快，但内部 tiling 和调度不在当前教学文件中展开。

**后续参考**：可参考 `reference/kernels/sgemm/`、`reference/kernels/hgemm/`、`reference/kernels/swizzle/` 和 `reference/kernels/ws-hgemm/`。建议拆成两条线：

- FP32 scalar 深化线：`8 x 8` thread tile 或更合理的 `BM/BN/BK/TM/TN` 参数 -> `float4` vectorized load/store -> shared memory layout/padding/swizzle -> double buffering -> `cp.async`（要求 `sm_80+`）。
- Tensor Core 线：FP16 input + FP32 accumulation 的 WMMA baseline -> shared memory staging + fragment layout -> `mma.sync m16n8k16` -> warp/MMA tiling -> multi-stage pipeline -> block swizzle / shared memory swizzle。`wgmma`、TMA、DSMEM 只作为 `sm_90+` 参考，不默认进入本项目实现。

---

## 🔜 下一步候选

下面步骤是路线参考，不代表已经实现或已经实测；实际学习时仍然每轮只引入一个主要优化手段。后续算子可以把已经学过的优化手段作为 baseline 组合使用；每个步骤只标出本轮真正新增的核心概念。

### Conv2d（二维卷积）
学习价值：理解卷积如何转化或映射成矩阵乘，并观察数据复用与边界处理。

参考状态：当前 `reference/` 暂无直接 Conv2d 主线；若开学，先独立实现 direct conv，再在 im2col/implicit GEMM 阶段复用 GEMM 路线。

推荐步骤：
1. naive direct conv：一个线程计算一个输出元素，先保证 padding/stride/dilation 正确
2. shared memory input tile：缓存输入 patch，减少相邻输出重复读取
3. constant / read-only filter：小卷积核权重走更适合广播或缓存的路径
4. im2col + GEMM：把卷积转成矩阵乘，复用 GEMM 优化经验
5. implicit GEMM：不显式落地 im2col，减少额外 HBM 读写
6. Winograd（小 kernel 可选）：用更多变换换取更少乘法，重点分析适用范围

### Attention（Q/K/V 分块 + online softmax）
学习价值：把 GEMM、Softmax 和分块访存融合起来，理解为什么 FlashAttention 能减少 HBM 往返。

参考实现：`reference/kernels/flash-attn/`、`reference/kernels/openai-triton/merge-attn-states/cuda_merge_attn_states.cu`。这些实现包含 MMA、shared Q/K/V、swizzle 和 split-Q/KV 等进阶路线；本项目开学时仍应先写可读的 CUDA C baseline。

推荐步骤：
1. naive attention：显式计算 `S = QK^T`、softmax、再乘 `V`
2. tiled QK：分块计算 score，建立 Q/K tile 的 shared memory 复用
3. online softmax per block：分块更新每行 max 和 sum，避免完整 `S` 落 HBM
4. fused `P * V` accumulation：softmax 结果不完整写回，直接累加输出
5. causal mask / padding mask：加入真实模型常见边界条件
6. split-Q / split-KV：处理长序列和并行度不足，必要时学习 partial attention state merge
7. shared memory layout / swizzle：减少 bank conflict，并观察 shared Q/K/V 的容量取舍
8. double buffering / Tensor Core（硬件支持时）：进一步提高 tile 计算吞吐

### Embedding / RoPE（模型前处理与位置编码）
学习价值：理解不规则 gather、连续向量拷贝和位置编码中的向量化访存。

参考实现：`reference/kernels/embedding/embedding.cu`、`reference/kernels/rope/rope.cu`。

推荐步骤：
1. embedding naive gather：按 token id 读取一整行 embedding，先处理索引边界
2. `float4` / 128-bit packed embedding copy：减少 load/store 指令并处理对齐约束
3. RoPE naive：实现偶偶/奇偶维度旋转和不同索引映射
4. RoPE `float4` packed：观察三角函数计算与访存指令的相对瓶颈

### Activation / Fused Elementwise（激活与融合）
学习价值：把已学的 elementwise 访存优化扩展到真实模型常见激活函数，并理解数学近似、dtype 和向量化的取舍。

参考实现：`reference/kernels/relu/`、`sigmoid/`、`gelu/`、`swish/`、`hardswish/`。

推荐步骤：
1. ReLU / Sigmoid / GELU naive：先建立 CPU reference 和误差阈值
2. `float4` packed FP32：复用 vec_add 的向量化访存经验
3. FP16 / `half2`：观察吞吐、误差和转换开销
4. fused bias + activation：减少一次 HBM 往返，重点分析 fusion 的有效访存收益

### Histogram / NMS（原子操作与不规则控制流）
学习价值：覆盖当前主线较少涉及的 atomic contention、分支发散和输出压缩问题。

参考实现：`reference/kernels/histogram/histogram.cu`、`reference/kernels/nms/`。

推荐步骤：
1. histogram naive atomic：建立 atomic contention 基线
2. block-local histogram：先在 shared memory 聚合，再写回 global memory
3. NMS CPU reference + naive CUDA：保证 IoU、排序输入和 keep mask 正确
4. bitmask / tiled NMS：减少重复比较，并分析分支发散和内存布局
