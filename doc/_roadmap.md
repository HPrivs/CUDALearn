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
4. two-pass reduction：第一轮写 partial sums，第二轮归约 partial，去掉跨 block 同地址 atomic 竞争
5. warp shuffle block reduce：warp 内用 `__shfl_down_sync` 归约，减少 shared memory 读写和部分同步

**当前瓶颈**：输入读取 + 额外 kernel launch 开销 + 少量 block 内归约开销

### Transpose（二维访存入门）
1. naive：一个线程搬运一个元素，读连续、写跨 stride
2. shared memory tiled transpose：用 shared memory 暂存 tile，交换 block 坐标后连续写回
3. shared memory padding：给 tile 行跨度加 1，减少转置读取时的 bank conflict

**当前瓶颈**：仍是 memory-bound；padding 在当前 GP108 上未稳定提速，下一步更适合转向 GEMV

### GEMV（矩阵×向量）
1. naive：一个线程计算一整行 dot product

**当前瓶颈**：memory-bound + 行内串行累加，下一步适合用 block 内并行 reduce

---

## 🔜 下一步候选

- **Softmax** — 行归约 + 指数归一化。引入 **online softmax**（单遍同时维护 max 和 sum）
- **LayerNorm / RMSNorm** — 归一化层。复合 reduce + elementwise，工程中高频
- **GEMM** — 矩阵乘法。CUDA 优化"终极 boss"，串联几乎所有手段：SMEM tile / register tile / double buffer / `cp.async` / Tensor Core / swizzle
- **Conv2d** — 二维卷积。im2col+GEMM / implicit GEMM / Winograd
- **Attention** — FlashAttention 式，Q/K/V 分块 + online softmax，S 不落 HBM（建议在 GEMM 和 Softmax 之后再学）
