# GEMM FP32 Scalar

## 学习目标

本文件新开一条不使用 Tensor Core 的 SGEMM 优化线，目标是把 FP32 scalar GEMM 中常见的 block/thread tile、`float4` packed load/store、shared memory layout/padding 和 software double buffering 放到一个可独立 benchmark 的文件里。

本轮新增概念不止一个，因为用户明确要求“FP32 scalar 的优化都用上”。为避免混在一起看不清收益，代码拆成 `v1/v2/v3` 三个版本逐步比较。

## 前置知识

- 已学：GEMM naive、shared memory tile、register tile、cuBLAS SGEMM 对照。
- 新术语：software double buffering 指用两份 shared memory 交替保存“当前计算 tile”和“下一轮预取 tile”，主循环里尝试让 global load 和 FFMA 计算交错。
- 本文件不学习 WMMA、`mma.sync`、Tensor Core、`wgmma` 或 TMA。

## 问题规格

- 数学定义：`C[M, N] = A[M, K] * B[K, N]`。
- 默认规模：`M = N = K = 1024`，与 `src/gemm.cu` 对齐。
- dtype：FP32 input、FP32 accumulation、FP32 output。
- layout：全部 row-major。
- correctness：CPU double accumulation reference，`atol=1e-3, rtol=1e-5`。

默认 `1024^3` 规模下：

| 口径 | bytes | FLOPs | AI |
| --- | ---: | ---: | ---: |
| naive 逻辑访存 | 8,594,130,944 | 2,147,483,648 | 0.25 |
| FP32 tiled 逻辑访存 | 71,303,168 | 2,147,483,648 | 30.12 |
| cuBLAS 最小有效访存 | 12,582,912 | 2,147,483,648 | 170.67 |

## naive — one thread per C

### 学习目标

建立正确性和性能下界：一个线程计算一个 `C[row, col]`，每个输出元素独立扫描 `K`。

### 定量分析

- 访存字节 `B`：每个输出读 `K` 个 `A`、`K` 个 `B`，写 1 个 `C`，默认约 `8.59 GB`。
- FLOPs `F`：每个输出 `2K` FLOPs，默认约 `2.15 GFLOPs`。
- `AI = F / B ≈ 0.25 FLOP/Byte`。
- 瓶颈判断：memory-bound，重复 global memory 读取非常严重。

## v1 — 8x8 thread tile

### 改了什么

`v1_8x8_tile` 使用 `BM=128, BN=128, BK=8, TM=8, TN=8`：

- 一个 block 计算一个 `128 x 128` 的 `C` tile。
- 一个线程计算 `8 x 8` 个输出元素。
- A/B 的 K tile 先放到 shared memory，再被 256 个线程复用。

### 为什么可能更快

`8 x 8` thread tile 让每个线程维护 64 个 FP32 accumulator。一次从 shared memory 读出的 `A` 或 `B` 值可以服务多个 FFMA，global memory 逻辑访存从 naive 的约 `8.59 GB` 降到约 `71.30 MB`。

### 代价或限制

64 个 accumulator 会明显提高 register pressure。这个版本仍用 scalar load/store，global memory 指令数量还可以继续减少。

### v1 作业

1. 推导题：默认 `BM=128, BN=128, BK=8` 时，为什么一个 `C` tile 每个 K tile 只需要从 global memory 读 `128*8 + 8*128` 个 float？
2. 判断题：`8 x 8` thread tile 为什么不一定比 `4 x 4` 永远更快？请从 register pressure 和 occupancy 两个角度回答。

## v2 — float4 + shared memory padding

### 改了什么

`v2_float4_pad` 在 v1 基础上引入：

- `float4` packed global load/store。
- A tile 在写 shared memory 时做 online transpose，布局为 `smem_a[BK][BM + PAD]`。
- B tile 保持 row-major shared memory layout：`smem_b[BK][BN + PAD]`。
- `PAD=4`，让 shared memory stride 不再正好是 128 个 float。

### 为什么可能更快

`float4` 让每次 global memory 指令搬 16 bytes，减少 load/store 指令数。A 的 shared memory online transpose 让计算阶段按 `K` 维读取时更直接，padding 则用额外 shared memory stride 改变 bank 映射，降低 bank conflict 风险。

### 定量分析

- 访存字节 `B`：逻辑 global traffic 和 v1 相同，默认约 `71.30 MB`。
- FLOPs `F`：仍是 `2.15 GFLOPs`。
- `AI ≈ 30.12 FLOP/Byte`。
- 瓶颈判断：更偏 compute/shared-memory 指令瓶颈；若 `float4` 对齐条件不满足，代码会退回 scalar 边界路径。

### v2 作业

1. 概念题：`float4` 优化为什么降低的是“指令条数”，而不是默认表格里的“逻辑访存字节”？
2. 改错题：如果把 `smem_a[BK][BM + PAD]` 改回 `smem_a[BM][BK]`，计算阶段的索引和 bank conflict 风险会发生什么变化？

## v3 — software double buffering

### 改了什么

`v3_double_buf` 使用两份 shared memory：

```text
smem_a[2][BK][BM + PAD]
smem_b[2][BK][BN + PAD]
```

主循环中一边把下一块 K tile 读入 `load_stage`，一边计算当前 `compute_stage`。这不是 `cp.async`，仍然是普通 global load；它只是通过两份 shared memory 和重排循环，减少每轮 load/compute 之间的同步阻塞。

### 为什么可能更快

v2 每轮 K tile 是：

```text
load -> sync -> compute -> sync
```

v3 先预取第一块，然后主循环变成：

```text
load next + compute current -> sync
```

同步次数减少，global load 指令也有机会和 FFMA 指令交错发射。实际能重叠多少取决于编译器调度、register pressure、occupancy 和硬件。

### 定量分析

- 访存字节 `B`：逻辑 global traffic 仍和 v1/v2 相同，默认约 `71.30 MB`。
- FLOPs `F`：仍是 `2.15 GFLOPs`。
- `AI ≈ 30.12 FLOP/Byte`。
- 瓶颈判断：compute-bound / shared-memory-latency-bound 混合。v3 不再主要靠减少 DRAM bytes，而是靠减少同步与改善指令调度。

可验证的 NCU metric 名已用 `ncu --query-metrics` 查询过，可重点看：

- `dram__bytes_read`
- `dram__bytes_write`
- `dram__throughput`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st`
- `smsp__inst_executed_pipe_fma`
- `smsp__sass_thread_inst_executed_op_ffma_pred_on`

### v3 作业

1. 解释题：v3 的 logical bytes 和 v2 相同，为什么实测仍可能更快？
2. 预测题：如果把 `BK` 从 8 改成 16，可能同时带来哪些收益和风险？

## cuBLAS SGEMM 对照

`cublas_sgemm` 使用 `cublasSgemm`，并设置 `CUBLAS_PEDANTIC_MATH`，尽量保持 FP32 scalar 口径。它是库函数基线，不属于手写 kernel 优化版本。

cuBLAS 的 `GB/s` 使用最小有效访存字节 `A+B+C`，不能和手写版本的逻辑 tiled bytes 直接比较；更适合比较 `ms` 和 `TFLOPS`。

## 对比总表

编译运行命令：

```bash
nvcc -O3 -arch=native src/gemm_fp32_scalar.cu -o debugger/gemm_fp32_scalar -lcublas && ./debugger/gemm_fp32_scalar
```

运行设备：NVIDIA GeForce RTX 4060 Ti，compute capability `8.9`。本轮为对齐 `src/gemm.cu`，两个文件都使用 `nvcc -O3 -arch=native` 编译，计时使用 `timeit` 默认 `warmup=3, iters=20`。

| version | ms | GB/s | TFLOPS | max_err |
| --- | ---: | ---: | ---: | ---: |
| naive | 1.5673 | 5483.28 | 1.3702 | 0.000021 |
| v1_8x8_tile | 0.2848 | 250.38 | 7.5410 | 0.000021 |
| v2_float4_pad | 0.2048 | 348.16 | 10.4858 | 0.000021 |
| v3_double_buf | 0.2448 | 291.29 | 8.7729 | 0.000021 |
| cublas_sgemm | 0.1801 | 69.88 | 11.9258 | 0.000004 |

阶段结论：

- `v1_8x8_tile` 相比 naive 快约 `5.50x`。
- `v2_float4_pad` 相比 v1 快约 `1.39x`。
- `v3_double_buf` 相比 v2 慢约 `1.20x`，说明普通 load 的 software double buffering 在当前配置下没有稳定收益。
- 最佳手写版本是 `v2_float4_pad`，相比 naive 快约 `7.65x`，逻辑 global traffic 减少约 `99.2%`。
- 当前 FP32 scalar 最佳版本达到本轮 cuBLAS FP32 pedantic baseline 的约 `87.9%`；对齐后明显快于 `src/gemm.cu` 的 `v5_final_4x4`。

同一轮对齐 `src/gemm.cu` 后，关键对比如下：

| file / version | ms | TFLOPS | 说明 |
| --- | ---: | ---: | --- |
| `src/gemm.cu` / `v5_final_4x4` | 0.3257 | 6.5943 | 简洁教学线最佳手写版本 |
| `src/gemm_fp32_scalar.cu` / `v2_float4_pad` | 0.2048 | 10.4858 | 当前 FP32 scalar 深化线最佳手写版本 |
| `src/gemm_fp32_scalar.cu` / `cublas_sgemm` | 0.1801 | 11.9258 | cuBLAS FP32 pedantic baseline |

用 `nvcc -O3 -arch=native --ptxas-options=-v src/gemm_fp32_scalar.cu -o /tmp/gemm_fp32_scalar_ptxas -lcublas` 查看资源占用：`kernel_v1` 使用 122 registers / 8192B shared memory，`kernel_v2` 使用 117 registers / 8448B shared memory，`kernel_v3` 使用 129 registers / 16896B shared memory。v3 继续提速的同时 register 和 shared memory 压力都更高，后续扩大 tile 或加 `cp.async` 前必须重新看 occupancy。

边界条件已用 `./debugger/gemm_fp32_scalar 257 263 269` 做 smoke test，所有版本通过 CPU reference。非整除小规模下优化版会因为边界分支和 tile 浪费变慢，这属于预期现象。

## 参考资料

- `reference/kernels/sgemm/sgemm.cu`
- `reference/kernels/sgemm/sgemm_async.cu`
- CUDA C++ Programming Guide：shared memory、global memory coalescing、thread hierarchy。
- cuBLAS Library：`cublasSgemm` column-major 调用约定。
