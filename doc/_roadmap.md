# 算子优化路径速查

已学部分详细记录优化步骤；未学部分作为下一步可选方向，仅列简介，开工后展开。

---

## ✅ 已学 / 进行中

### Elementwise / VecAdd（入门首选）
1. naive：一个线程处理一个元素

**瓶颈演进**：始终 memory-bound，目标逼近 DRAM 带宽峰值

### Reduce / Sum（入门第二步）
1. naive：atomicAdd 到全局变量（正确但极慢）
2. SMEM 树形归约：block 内 shared memory 折半相加
3. warp shuffle：`__shfl_xor_sync` / `__shfl_down_sync` 免 SMEM
4. 两阶段归约：block 内归约 → grid 再归约一次，避免 atomic

**瓶颈演进**：memory-bound → latency-bound（同步开销主导）

---

## 🔜 下一步候选（尚未展开，按难度递增）

- **Transpose** — 矩阵转置。入门经典，引出 **bank conflict** 和 SMEM padding 技巧
- **GEMV** — 矩阵×向量。Reduce 的自然延伸，开始接触 2D tiling
- **Softmax** — 行归约 + 指数归一化。引入 **online softmax**（单遍同时维护 max 和 sum）
- **LayerNorm / RMSNorm** — 归一化层。复合 reduce + elementwise，工程中高频
- **GEMM** — 矩阵乘法。CUDA 优化"终极 boss"，串联几乎所有手段：SMEM tile / register tile / double buffer / `cp.async` / Tensor Core / swizzle
- **Conv2d** — 二维卷积。im2col+GEMM / implicit GEMM / Winograd
- **Attention** — FlashAttention 式，Q/K/V 分块 + online softmax，S 不落 HBM（建议在 GEMM 和 Softmax 之后再学）

---

## 通用优化原则（入门 7 条）
1. **正确性第一**：每版必做 `max_abs_err` 对比，错了再快也没用
2. **先找瓶颈再动手**：memory-bound 就别死磕计算优化
3. **算术强度** `AI = FLOPs / Bytes` 是最重要的单个指标；低就想办法复用数据
4. **访存合并**：相邻线程访问相邻地址，才能一次事务完成
5. **用 SMEM 复用数据**：多次用到同一块数据就搬到 shared memory
6. **减少同步**：`__syncthreads` 越少越好，能用 warp 级就别用 block 级
7. **先测再猜**：`timeit` + 对比表是判断优化是否有效的唯一依据

---

<!-- 算子学习顺序说明 -->
<!-- 开始学某算子时：把对应条目从「🔜 下一步候选」移到「✅ 已学」，展开详细步骤 -->
<!-- 发现新候选时：追加到「🔜 下一步候选」末尾 -->
