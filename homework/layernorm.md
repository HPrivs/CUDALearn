# layernorm 作业

## 使用说明
- 在「我的答案」下面写你的推导、代码片段或解释。
- 「自我检查」用于你自己记录不确定点。
- 需要我批改时，直接说「批改 layernorm 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、求 `mean`、求 `var`、计算 `inv_std`、写回 `y[row, col]`。

### 我的答案

``` cpp
__global__ void kernel_naive(const float* x, float* y, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const size_t row_offset = static_cast<size_t>(row) * cols;

    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum += x[row_offset + col];
    }
    const float mean = sum / static_cast<float>(cols);

    float sq_sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        const float diff = x[row_offset + col] - mean;
        sq_sum += diff * diff;
    }
    const float var = sq_sum / static_cast<float>(cols);
    const float inv_std = 1.0f / sqrtf(var + kEps);

    for (int col = 0; col < cols; ++col) {
        y[row_offset + col] = (x[row_offset + col] - mean) * inv_std;
    }
}
```

### 自我检查


### 批改反馈

正确。

关键要点：
- `row = blockIdx.x * blockDim.x + threadIdx.x` 和 `row >= rows` 边界检查都写对了。
- `row_offset = row * cols` 写对，后续三次循环都基于这一行的连续区间访问。
- `mean`、`var`、`inv_std`、最终写回公式都完整，且 `sqrtf(var + kEps)` 的 float kernel 写法正确。

标准思路：naive LayerNorm 就是一个线程处理一整行，按 `sum -> mean -> sq_sum -> var -> inv_std -> normalize` 的顺序做三次行扫描。你这版没有漏掉边界检查，也没有把 `var` 写成除以 `cols - 1`，可以作为正确的默写答案。


### 题目 2
用一段话解释：为什么 naive LayerNorm 的有效 `GB/s` 很低？请至少从“每行并行度”和“warp 内访存是否连续”两个角度回答。

### 我的答案

每行并行度只有1，单行的大量数据都是串行执行的，这是最主要的点。
warp内访存上看，warp内线程访存地址的间隔是sizeof(float) * cols。远远超出了连续合并访存的要求。


### 自我检查

两点没有从现象-硬件后果具体反映到为什么`GB/s`很低。
每行并行度低，整体可调度的总block数也不多（rows / kblocksize），不能用多个线程做行内规约。
warp的相邻线程处理不同行，不是连续global memory access，结果是并行度不足/memory transaction效率差、latency难隐藏。


### 批改反馈

基本正确。

关键要点：
- “每行并行度只有 1”说到了核心：`K = 4096` 时，一个线程要串行扫完整行三遍，行内没有并行归约。
- “warp 内线程访存地址间隔是 `sizeof(float) * cols`”判断正确。row-major 下，同一 warp 的相邻线程处理相邻行，在同一个 `col` 上访问的地址相差整行长度，因此不满足连续合并访存。
- 可以再补一句：虽然逻辑访存量看起来是三读一写，`AI` 很低，但 naive 版不是单纯 memory bandwidth 打满后的 memory-bound，而是同时受行内串行、warp 访存不合并和 long scoreboard latency 影响，所以有效 `GB/s` 会很低。

标准思路：naive 版把一行交给一个线程，导致行内计算和三次读 `x` 都串行执行；同时同一个 warp 的线程分别处理不同行，访问地址跨 `cols * sizeof(float)`，global load 难以 coalesce。结果是并行度和访存形态都差，DRAM 带宽没有被充分利用。


## v2 作业

### 题目 1
为什么 v2 的逻辑访存字节 `B` 和 v1 一样，但实测 `GB/s` 和耗时明显改善？请从行内并行度和 memory coalescing 两个角度回答。

### 我的答案


### 自我检查


### 批改反馈


### 题目 2
下面这段 v2 风格代码有两个和 block-per-row 映射相关的问题，请指出并说明后果。

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const size_t row_offset = static_cast<size_t>(row) * cols;

float thread_sum = 0.0f;
for (int col = 0; col < cols; col += blockDim.x) {
    thread_sum += x[row_offset + col];
}
```

### 我的答案


### 自我检查


### 批改反馈
