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

指数范围更准确是(0, 1]。公共因子是exp(-m)此处是笔误。

### 批改反馈

基本正确。减去最大值后，每个 `z_i - m <= 0`，所以 `exp(z_i - m)` 落在 `(0, 1]`，最大那个元素对应的指数值是 `1`，这样能避免 `exp(z_i)` 因输入太大而 overflow。

需要修正一处推导表述：公共因子应写成 `exp(-m)`，不是 `exp(z_m)`。因为 `exp(z_i - m) = exp(z_i) * exp(-m)`，分子和分母同时乘了同一个 `exp(-m)`，所以会约掉，softmax 结果不变。你的“平移不变性”结论是对的。

## v2 作业

### 题目 1
解释 v2 为什么比 v1 快：请分别从“每行并行度”“逻辑访存量”“同步开销”三个角度回答。

### 我的答案


### 自我检查


### 批改反馈


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


### 自我检查


### 批改反馈
