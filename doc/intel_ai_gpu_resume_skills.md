# Intel AI GPU 架构性能优化实习生简历技能提炼

本文档基于当前 `CUDALearn` 项目已有代码、算子优化文档，以及 `doc/note/CUDA Programming Guide/` 学习笔记整理，目标是把项目经历提炼成适合「AI GPU 架构性能优化实习生」岗位的专业技能表述。简历中应优先使用已在项目中实现、测试或明确分析过的内容；仅来自学习笔记、尚未形成项目实作的内容，应写成「理解 / 了解」，不要写成「熟练工程实践」。

## 简历技能栏精简版

下面这版适合直接放进简历「专业技能」栏目，按能力块合并，避免过散。

- 熟悉 C/C++ 与 CUDA C++ 编程，掌握指针、数组、函数、模板基础和 STL 常用容器；能够编写 CUDA kernel、CPU reference、benchmark 以及错误检查和结果校验代码。
- 熟悉 CUDA kernel 开发与性能优化流程，实践过向量类、归约类、矩阵类和归一化类等基础 AI / 数值计算算子，能够进行多版本 correctness 验证、benchmark 对比和瓶颈分析。
- 熟悉 GPU 并行计算与内存层次基础，理解 CUDA 线程组织、SM 与 warp 执行模型、全局内存与共享内存访存模式、寄存器压力与 occupancy 取舍；熟悉 shared memory tiling、register tiling、warp-level reduction、two-pass reduction、vectorized load/store 等常见优化方法。
- 了解 Roofline model 的基本思想，能够结合访存量、计算量、算术强度、有效带宽和计算吞吐判断 kernel 的 memory-bound / compute-bound 特征，并进一步分析同步开销、访存模式和数值误差对性能与正确性的影响。
- 熟悉 Python 与 PyTorch 基础使用，了解张量计算、自动求导、模型构建、数据加载、训练与推理流程；了解 CNN / Transformer 中常见层和算子。
- 了解 AI 算子的数值稳定性、低精度计算与硬件特性映射，实践过 online softmax、Welford 方差计算、RMSNorm，并尝试过基于 Tensor Core 的低精度矩阵乘路径。
- 了解 CUDA Runtime、内存管理与编译工具链基础，理解异步执行、页锁定内存、统一内存、统一虚拟寻址、编译产物和目标架构等概念。

## 项目证据映射

| 能力方向 | 当前项目证据 |
| --- | --- |
| CUDA kernel 正确性与 benchmark | `src/common.cuh` 提供 `CUDA_CHECK`、`timeit`、`random_fill`、`max_abs_err`、`print_row`；各 `src/X.cu` 包含 CPU reference、版本循环、correctness 和性能输出 |
| CUDA 编程模型基础 | `doc/note/CUDA Programming Guide/1.1`、`2.1`、`2.2` 笔记覆盖 host/device、grid/block/thread、SM/GPC、warp/SIMT、warp divergence、线程线性化、边界检查 |
| 设备内存空间与缓存 | `doc/note/CUDA Programming Guide/2.2` 笔记覆盖 global/shared/local/register/constant memory、L1/L2 cache、shared memory 静态/动态分配、local memory spill |
| 访存与 bandwidth 分析 | `Elementwise` 使用 `float4` vectorized load/store，文档说明 `B/F/AI` 不变但指令组织改善；`Transpose` 对比跨 stride 写、shared memory tile 和 padding |
| Reduction 优化 | `Reduce` 从 global atomic 到 shared memory reduce、register accumulation、two-pass、warp shuffle，性能从 `31.5855 ms` 优化到 `1.3732 ms` |
| Row-wise AI kernel | `Softmax` 从单线程三次扫描优化到 block-per-row、warp shuffle、online softmax，性能从 `104.8098 ms` 优化到 `4.5306 ms` |
| 归一化层数值与性能权衡 | `LayerNorm` 覆盖 naive、block+warp reduce、Welford variance、RMSNorm，分析减少一次读取与新增算术/reduction 开销之间的取舍 |
| GEMM 数据复用与 tile 设计 | `GEMM` 覆盖 shared memory tile、`2 x 1`、`2 x 2`、`4 x 4` register tile，FP32 版本从 `4.4468 ms / 0.0604 TFLOPS` 优化到 `0.5058 ms / 0.5307 TFLOPS` |
| 低精度与硬件路径 | `GEMM v6` 使用 FP16 input、FP32 accumulation/output、WMMA fragment、`cp.async` double buffer，并标注 `sm_80+` 实测前提 |
| 异步执行与运行时 | `doc/note/CUDA Programming Guide/2.3` 笔记覆盖 stream/event、`cudaMemcpyAsync`、pinned memory、默认流、非阻塞流、跨流同步、异步错误处理、CUDA Graphs |
| 内存管理与迁移 | `doc/note/CUDA Programming Guide/2.4`、`4.1` 笔记覆盖 UVA、Unified Memory、HMM/ATS、page-locked host memory、mapped memory、`cudaMemAdvise`、`cudaMemPrefetchAsync` |
| 编译工具链 | `doc/note/CUDA Programming Guide/2.5` 笔记覆盖 `nvcc`、PTX、Cubin、Fatbin、`-arch`、`-gencode`、`-rdc=true`、CUDA runtime linking |
| 分析边界意识 | 多个文档明确区分「已实测」「未实测」「需要 profiler 验证」，避免把 `GB/s`、`TFLOPS` 或 NCU metric 当作脱离硬件的绝对结论 |

## 面试中可展开的技术点

- 为什么 `float4` 不减少理论访存字节，却可能提升 bandwidth。
- 为什么 `Reduce` 中全局同地址 `atomicAdd` 会成为严重瓶颈，two-pass reduction 如何去掉跨 block atomic contention。
- 为什么 `Transpose` 需要 shared memory tile，以及 padding 如何缓解 shared memory bank conflict。
- 为什么 `GEMV` 的 `AI` 很低，但 block-per-row 仍能明显加速。
- 为什么 `Softmax` 和 `LayerNorm` 不能只看 FLOPs，需要考虑行内并行度、`expf` / `sqrtf` 延迟、reduction 和 global memory scan 次数。
- 为什么 GEMM 的 register tile 可以提高数据复用，但过大的 tile 会带来 register pressure、occupancy 下降和 spill 风险。
- Tensor Core / `cp.async` 路径和 FP32 scalar GEMM 的差异：dtype、warp-level MMA、fragment layout、global-to-shared pipeline 和硬件前提。
- 为什么同一 thread block 内可以用 shared memory 和 `__syncthreads()` 协作，但不同 block 之间通常不能依赖执行顺序。
- 为什么 `cudaMemcpyAsync` 要配合 pinned host memory 才能真正异步，默认流为什么可能破坏跨流并发。
- Unified Memory 解决的是编程便利性，不代表数据不迁移；mapped memory 访问主机内存通常受 PCIe / NVLink 链路延迟和带宽限制。
- `nvcc` 中 `compute_XX`、`sm_XX`、PTX、Cubin、Fatbin 的区别，以及为什么硬件专属优化必须标注架构前提。

## 建议避免的过度表述

- 不建议写「熟悉 Intel GPU / SYCL / OpenCL 性能优化」，当前项目主要是 CUDA 路径；可以写「具备 GPU kernel 优化基础，愿意迁移到 SYCL/OpenCL/Intel GPU 工具链」。
- 不建议写「熟练使用 Nsight Compute 定位瓶颈」，当前项目更多是列出可验证 metric 和部分报告文件，适合写「具备 profiler 指标验证意识」。
- 不建议写「精通 Tensor Core / `cp.async` 高性能 GEMM」，当前 v6 已交叉编译并检查 SASS，但尚未在 `sm_80+` 实机 benchmark。
- 不建议写「熟练使用 CUDA Graphs / Unified Memory 做性能优化」，目前它们主要来自 Programming Guide 笔记，尚未在项目算子中做系统 benchmark。
- 不建议写「掌握稀疏计算和量化感知优化」，当前项目尚未实现 sparse kernel 或 quantization-aware workload；可以在兴趣方向中表达后续计划。
