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

1、行内并行度从1提高为了kBlockSize=256，行内进行了warp-level tree reduce，大幅减少了每个线程串行循环长度；同时每行由一个block负责，提高了总block数量增加了有效`GB/s`。
2、warp内相邻线程访问的地址也是相邻的，global load满足memory coalescing条件，充分利用了DRAM带宽。

### 自我检查

1的结论这里，逻辑是`线程分摊->每个线程串行循环长度下降->更多线程能同时发起访存和计算->同样的逻辑字节能在更短时间完成`。

### 批改反馈

真正改善的是执行方式：v1 中一行由一个线程串行扫完整行；v2 中一行由一个 block 的 `kBlockSize = 256` 个线程分摊，每个线程每轮大约处理 `cols / 256` 个元素，再通过 block/warp reduce 合并结果。这样每个线程的串行循环长度下降，更多线程能同时发起访存和计算，同样的逻辑字节能在更短时间内完成，所以有效 `GB/s = B / time` 会提高。

memory coalescing 的回答是对的：v2 中同一 warp 的相邻线程在第一次迭代访问 `x[row_offset + tid]` 这类连续地址，更容易合并成连续 global memory transaction；v1 则是相邻线程处理不同行，在同一 `col` 上地址相隔 `cols * sizeof(float)`，访问形态更差。你提到 block 数增加也有一定道理，但主因仍是行内并行度和连续访存，而不是 `B` 变小。

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

block-per-row映射条件下的row应该是`row = blockIdx.x`，每个线程块处理一行。上面的row是基于每个线程处理一行的线性映射，放在block-per-row会导致DRAM访存错乱。

循环条件中col的初始值应该是`int col = tid`，这种循环初始化会使得每个线程都访问同一处DRAM地址，导致DRAM访存未能被完整覆盖。


### 自我检查

第一个问题更准确的说是`block内线程分散到不同行，导致不同行的partial sum混在一起`。

### 批改反馈

正确，两个问题都定位到了。

第一个 bug 是 `row` 仍然用了 one-thread-per-row 的线性线程编号。block-per-row 中应该写成 `row = blockIdx.x`，否则同一个 block 内 256 个线程会分散到 256 个不同的行。后果不是简单的“访存错乱”，而是 block 内归约会把不同 row 的 partial sum 混在一起，得到没有数学意义的 mean/variance；如果仍按 `kernel_v2<<<rows, kBlockSize>>>` 启动，还会让 `row` 最大到 `rows * blockDim.x - 1`，产生大量越界访问。

第二个 bug 是列循环应该从 `threadIdx.x` 开始：`for (int col = tid; col < cols; col += blockDim.x)`。如果从 `0` 开始，在正确的 block-per-row 映射下，block 内所有线程会重复读取同一批列 `0, blockDim.x, 2 * blockDim.x, ...`，既没有覆盖整行，也浪费了并行线程；在当前错误的 `row` 映射下，则会变成不同线程访问不同 row 的同一组列，访问不连续且结果仍然错误。

## v3 作业

### 题目 1
v3 的逻辑访存字节比 v2 少 25%，为什么实测耗时却几乎不变？请从 DRAM 读减少、Welford per-element arithmetic、Welford combine 三个角度回答。

### 我的答案


### 自我检查


### 批改反馈


### 题目 2
下面的 Welford partial 合并方式有什么问题？为什么不能这样合并两个 partial statistics？

```cpp
WelfordData wrong_combine(WelfordData a, WelfordData b) {
    WelfordData out;
    out.count = a.count + b.count;
    out.mean = a.mean + b.mean;
    out.m2 = a.m2 + b.m2;
    return out;
}
```

### 我的答案


### 自我检查


### 批改反馈
