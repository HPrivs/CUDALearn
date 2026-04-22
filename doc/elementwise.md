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
nvcc -O3 -std=c++17 src/elementwise.cu -o elementwise && ./elementwise
```

把输出表里的 `ms / GB/s / max_err` 记下来，后面 v2、v3 都要和它比。

### 瓶颈
这个版本的瓶颈不是计算，而是 HBM / DRAM 带宽。每个元素只做 1 次加法，却要读 2 个 `float`、写 1 个 `float`。

### 下一步
下一版最自然的方向是 **向量化访存**：让每个线程一次搬 16B，例如 `float4`，减少指令条数并改善访存效率。

## 对比总表
| version | ms | GB/s | TFLOPS | 说明 |
| --- | --- | --- | --- | --- |
| naive | 待实测 | 待实测 | 待实测 | 一个线程处理一个元素 |

## 作业
### v1 作业
1. 合上代码，自己默写 `kernel_naive` 的核心三行：算 `idx`、做边界检查、完成 `c[idx] = a[idx] + b[idx]`。
2. 解释为什么这个算子的 `AI = 1/12` 很低，并用一句话判断它更像 memory-bound 还是 compute-bound。
