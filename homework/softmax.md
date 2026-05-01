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

global memory 扫描仍然是三次，比起v2，DRAM的访存并没有改变；
`expf`仍然要进行两次`expf(x[row_offset + col] - max_val)`；
v3更快的地方在shared memory reduce，引入了warp shuffle，减少了block-level同步开销和shared memory的访存开销。


### 自我检查


### 批改反馈

正确。

你抓住了 v3 只小幅快于 v2 的核心原因：v3 优化的是 block 内 reduce 的实现方式，而不是 softmax 的主要外层工作量。global memory 仍然是三次扫描 `x`、一次写 `y`，所以有效 DRAM 访存量没有下降；`expf` 仍然在 sum 阶段和 write 阶段各算一次，也没有减少 special function 的计算压力。

更精确地说，v3 的收益来自把 v2 中 warp 内的 shared memory tree reduce 换成 register-level `__shfl_down_sync`，因此减少 shared memory load/store 和部分 `__syncthreads()` 带来的 barrier 开销。但默认 `K = 4096` 时，每行还有大量 global load 和两轮 `expf`，reduce 开销只占总时间的一部分，所以 benchmark 里只看到约 `1.04x` 的小幅提升是合理的。


### 题目 2
改错题：如果把 `block_reduce_sum` 里非 partial 线程的单位元写成 `-FLT_MAX`，会造成什么错误？为什么 max reduce 和 sum reduce 的单位元不同？

### 我的答案

`block_reduce_max`里非partial线程的单位元使用`-FLT_MAX`是为了防止非partial线程干扰比较结果。在`block_reduce_sum`使用`-FLT_MAX`单位元会导致规约时partial线程的val加上干扰值`-FLT_MAX`导致结果错误；
单位元不同是因为`-FLT_MAX`是用于比较的；`0.0f`是用于求和的。

### 自我检查


### 批改反馈

正确。

`block_reduce_sum` 里只有 `tid < num_warps` 的线程持有真实的 warp partial sum，其余线程只是为了凑满第一个 warp 的归约形状，必须放入“不会改变求和结果”的单位元，也就是 `0.0f`。如果这些线程写成 `-FLT_MAX`，sum reduce 会把这些巨大负数加进 `denom`，轻则得到错误的负数或极端值，重则后续 `exp(...) / denom` 产生完全错误的 softmax 输出。

max reduce 和 sum reduce 的单位元由运算本身决定：`max(x, -FLT_MAX) = x`，所以 `-FLT_MAX` 适合 max；`x + 0 = x`，所以 `0.0f` 适合 sum。这里不能因为两个函数结构相似就复用同一个单位元。

## v4 作业

### 题目 1
概念题：给定旧状态 `(m_old, d_old)` 和新元素 `x`，分别写出 `x <= m_old` 与 `x > m_old` 两种情况下 online softmax 的更新公式，并解释为什么最大值变大时旧的 `d_old` 必须乘 `exp(m_old - x)`。

### 我的答案
伪代码：
``` cpp
if (x > m_old) {
    m_new = x;
    d_new = d_old * exp(m_old - x) + 1.0f;
}
else {
    d_new = d_old + exp(x - m_old); 
}
```

乘`exp(m_old - x)`首先是为了将分母各项的指数还原回未减去`m_old`的状态再减去新最大值`x`。

### 自我检查

解释上，这里的缩放用还原不太合适，应该说把旧分母从`m_old`基准转换到新最大值`x`的基准。

### 批改反馈

基本正确。

公式部分只差一个小补全：当 `x <= m_old` 时，最大值不变，所以应明确写成 `m_new = m_old`，`d_new = d_old + exp(x - m_old)`。当 `x > m_old` 时，你写的 `m_new = x`，`d_new = d_old * exp(m_old - x) + 1.0f` 是对的。

解释也抓住了核心。更精确地说，旧的 `d_old` 定义在旧最大值基准上：

```text
d_old = sum_i exp(x_i - m_old)
```

当新最大值变成 `x` 后，旧元素必须改成新基准：

```text
sum_i exp(x_i - x)
= sum_i exp(x_i - m_old) * exp(m_old - x)
= d_old * exp(m_old - x)
```

最后再加上新元素自己的贡献 `exp(x - x) = 1`。这个缩放不是为了“还原回未减去 `m_old` 的状态”本身，而是为了把旧分母从 `m_old` 基准转换到新最大值 `x` 基准。

### 题目 2
改错题：如果合并两个 partial state 时直接写 `d = d_a + d_b`，但 `m_a != m_b`，会造成什么错误？写出正确的合并公式。

### 我的答案

会导致分母中`d_a`或`d_b`其中一方的指数减去的并非最大值，造成数值错误。
正确的合并公式：
`m_a > m_b:`
`d = d_a + d_b * exp(m_b - m_a)`
`m_b >= m_a:`
`d = d_a * exp(m_a - m_b) + d_b`

### 自我检查


### 批改反馈

正确。

问题本质是 `d_a` 和 `d_b` 不是同一个尺度下的分母：`d_a` 是按 `m_a` 归一化得到的，`d_b` 是按 `m_b` 归一化得到的。若 `m_a != m_b`，直接 `d_a + d_b` 等于把两个不同 max 基准下的指数和硬加在一起，分母会偏大或偏小，最终 softmax 概率整体错误。

你的分支公式是对的：

```text
if m_a > m_b:
    m = m_a
    d = d_a + d_b * exp(m_b - m_a)
else:
    m = m_b
    d = d_a * exp(m_a - m_b) + d_b
```

也可以写成统一形式：`m = max(m_a, m_b)`，`d = d_a * exp(m_a - m) + d_b * exp(m_b - m)`。这个统一写法更不容易漏掉某一侧的缩放。
