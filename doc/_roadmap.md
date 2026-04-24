# CUDA 算子学习路线

## ✅ 已学 / 进行中

### Elementwise / VecAdd（入门首选）
1. naive：一个线程处理一个元素
2. `float4` 向量化访存：一个线程处理 4 个连续元素，减少访存指令条数

**瓶颈演进**：始终 memory-bound，目标逼近 DRAM 带宽峰值

### Reduce / Sum（入门第二步）
1. naive：atomicAdd 到全局变量（正确但极慢）
2. shared memory block reduce：每个 block 先做局部归约，再执行一次全局 atomicAdd
3. per-thread accumulation：每个线程先在 register 中累加多个元素，再进入 shared memory 归约

**当前瓶颈**：输入读取 + block 内同步开销 + shared memory 归约开销 + 剩余全局 atomic contention

---

## 🔜 下一步候选

- **Transpose** — 矩阵转置。入门经典，引出 **bank conflict** 和 SMEM padding 技巧
- **GEMV** — 矩阵×向量。Reduce 的自然延伸，开始接触 2D tiling
- **Softmax** — 行归约 + 指数归一化。引入 **online softmax**（单遍同时维护 max 和 sum）
- **LayerNorm / RMSNorm** — 归一化层。复合 reduce + elementwise，工程中高频
- **GEMM** — 矩阵乘法。CUDA 优化"终极 boss"，串联几乎所有手段：SMEM tile / register tile / double buffer / `cp.async` / Tensor Core / swizzle
- **Conv2d** — 二维卷积。im2col+GEMM / implicit GEMM / Winograd
- **Attention** — FlashAttention 式，Q/K/V 分块 + online softmax，S 不落 HBM（建议在 GEMM 和 Softmax 之后再学）
