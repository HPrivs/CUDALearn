# softmax 作业

## 使用说明
- 在「我的答案」下面写你的推导、代码片段或解释。
- 「自我检查」用于你自己记录不确定点。
- 需要我批改时，直接说「批改 softmax 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、求 `max_val`、求 `denom`、写回 `y[row, col]`。

### 我的答案

``` cpp

__global__ void kernel_naive(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const size_t row_offset = static_cast<size_t>(row) * cols;

    float max_val = -FLT_MAX;
    for (int col = 0; col < cols; ++col) {
        max_val = fmaxf(max_val, x[row_offset + col]);
    }

    float denom = 0.0f;
    for (int col = 0; col < cols; ++col) {
        denom += expf(x[row_offset + col] - max_val);
    }

    for (int col = 0; col < cols; ++col) {
        y[row_offset + col] = expf(x[row_offset + col] - max_val) / denom;
    }
}

```

### 自我检查


### 批改反馈

正确。核心四步都写到了：计算 `row`、用 `row >= rows` 排除越界线程、三次扫描同一行分别求 `max_val`、求 `denom`、写回输出。

一个值得保留的点：你把 `max_val` 初始化为 `-FLT_MAX` 是正确写法。这样即使一整行输入都是负数，也能得到正确最大值；如果初始化成 `FLT_MIN`，它表示最小正正规 float，不是“最负的 float”，遇到全负输入会出错。

### 题目 2
用一段话解释：为什么 softmax 要先减去每行最大值？这个操作为什么不会改变 softmax 的结果？

### 我的答案

1、减去最大值是为了控制exp(z_i)值域在[0, 1]，保持整体数值稳定。2、不改变结果是因为softmax具有平移不变性。softmax(z) = exp(z_i) / \sum(exp(z_k))；在此式上减去最大值m = z_j: softmax(z-m) = exp(z_i - m) / \sum(exp(z_k - m))，可提取公共因子exp(z_m)并约去，有：softmax(z-m) = softmax(z)。

### 自我检查

指数范围更准确是`(0, 1]`。公共因子是exp(-m)此处是笔误。

### 批改反馈

基本正确。减去最大值后，每个 `z_i - m <= 0`，所以 `exp(z_i - m)` 落在 `(0, 1]`，最大那个元素对应的指数值是 `1`，这样能避免 `exp(z_i)` 因输入太大而 overflow。

需要修正一处推导表述：公共因子应写成 `exp(-m)`，不是 `exp(z_m)`。因为 `exp(z_i - m) = exp(z_i) * exp(-m)`，分子和分母同时乘了同一个 `exp(-m)`，所以会约掉，softmax 结果不变。你的“平移不变性”结论是对的。

## v2 作业

### 题目 1
解释 v2 为什么比 v1 快：请分别从“每行并行度”“逻辑访存量”“同步开销”三个角度回答。

### 我的答案

v2版本的每行并行度从1提升到了kBlockSize=256，提升了每行并行度，在softmax这个需要大量串行工作的kernel下，大大降低了线程每行串行的工作量。
v2的逻辑访存量不变，仍然是读3次x写1次y（M * K * 4）。片上访存增加了smem[kBlockSize]的多次访存，包括树形查找最大值和树形规约。
v2的同步开销是增加了的，因为大部分写入smem[tid]情况都需要等待规约结束。

### 自我检查

逻辑访存量漏了sizeof(flaot) = 4，所以逻辑访存量应该是`M * K *16`。

### 批改反馈

基本正确。

关键点都覆盖到了：v2 把每行从 1 个线程提升到一个 block 内 `kBlockSize = 256` 个线程，单个线程负责的列数大约从 `K` 降到 `K / 256`，这是主要加速来源；你也正确指出 v2 的 global memory 逻辑访存模式没有减少，仍然是三次读 `x`、一次写 `y`；同步和 shared memory 访问是 v2 新增代价。

需要修正一个数字：逻辑访存量如果按字节算，应是 `M * K * 4 * sizeof(float) = M * K * 16 bytes`，不是 `M * K * 4`。这里的 `4` 是“3 次读 + 1 次写”的访问次数，乘上 `sizeof(float)` 后才是字节数。

“大部分写入 `smem[tid]` 情况都需要等待规约结束”这句可以说得更精确：每次把局部结果写入 `smem[tid]` 后，需要一次同步保证所有线程都写完；reduce 每一轮之后也需要同步，保证下一轮读到的是上一轮已经合并好的结果。


### 题目 2
改错题：下面这段 reduce 少了一处关键同步。指出 bug 在哪里，并说明可能造成什么错误。

```cpp
smem[tid] = thread_max;
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
}
```

### 我的答案

smem[tid] = thread_max;下方需要一次同步，不然可能会读取未完成写入的smem[tid]。

### 自我检查


### 批改反馈

正确。缺少的就是：

```cpp
smem[tid] = thread_max;
__syncthreads();
```

原因是所有线程必须先把自己的 `thread_max` 写入 `smem[tid]`，后续 reduce 才能安全读取 `smem[tid + stride]`。如果少了这次同步，部分线程可能还没写完，另一些线程已经开始读 shared memory，结果会读到旧值或未初始化值，导致 `max_val` 错误。`max_val` 一错，后面的 `denom` 和最终 softmax 输出都会错。

## v3 作业

### 题目 1
解释为什么 v3 只比 v2 小幅更快：请从 global memory 扫描、`expf`、shared memory reduce 三个角度回答。

### 我的答案


### 自我检查


### 批改反馈


### 题目 2
改错题：如果把 `block_reduce_sum` 里非 partial 线程的单位元写成 `-FLT_MAX`，会造成什么错误？为什么 max reduce 和 sum reduce 的单位元不同？

### 我的答案


### 自我检查


### 批改反馈
