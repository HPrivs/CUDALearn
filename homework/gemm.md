# GEMM 作业

## 使用说明
- 在「我的答案」下面写你的推导、代码片段或解释。
- 「自我检查」用于你自己记录不确定点。
- 需要我批改时，直接说「批改 gemm 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row/col`、边界检查、循环 `kk` 做 dot product、写回 `C[row, col]`。

### 我的答案

```cpp
__global__ void kernel_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m && col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        sum += a[static_cast<size_t>(row) * k + kk] * b[static_cast<size_t>(kk) * n + col];
    }
    c[static_cast<size_t>(row) * n + col] = sum;
}

```
### 自我检查

边界检查有所疏忽条件应该是`row >= rows || col >= cols`

### 批改反馈

结论：大部分正确，但边界检查有一个关键 bug。

- `row/col` 的计算、`kk` 循环、`A[row, kk] * B[kk, col]`、写回 `C[row, col]` 都写对了。
- 边界检查应为 `if (row >= m || col >= n) return;`，不能用 `&&`。只要 `row` 或 `col` 任意一个越界，就不能继续访问 `A/B/C`。
- 现在的 `&&` 只有在 `row` 和 `col` 同时越界时才返回；如果 `row >= m` 但 `col < n`，会读越界的 `A[row * k + kk]` 并写越界的 `C[row * n + col]`。如果默认规模刚好能被 `16x16` 整除，这个 bug 不一定暴露，但换成非整除尺寸就会出错。


### 题目 2
用一段话解释：为什么 naive GEMM 没有利用好数据复用？请分别说明 `A[row, kk]` 和 `B[kk, col]` 在一个输出 tile 内本来可以怎样被复用。

### 我的答案

在C的同一行中，计算每一列的元素都会使用同一个row的A，但是代码中每次都会从DRAM中访问A[row, kk]，这是可以被缓存复用用。
相同的，C的同一列计算也都会使用同一个col的B，但仍然从DRAM频繁访问B[kk, col]。

在一个输出tile内，例如block(16,16)，应该缓存A对应的16行数据和B对应的16列数据。这样一个tile的访存就从2K * TILE * TILE变成了2K * TILE。

### 自我检查

不应该说是tile的访存而是tile的global load，C的写是不变的。


### 批改反馈

结论：方向正确，已经抓住了 naive GEMM 的主要问题：同一个 tile 内本来有大量 `A/B` 复用，但 naive 让不同线程各自从 global memory 读。

关键要点：

- 对固定的 `row` 和 `kk`，`A[row, kk]` 会被同一输出 tile 中这一行的多个 `col` 使用；如果 tile 宽度是 `TILE_N=16`，它理论上可被 16 个输出元素复用。
- 对固定的 `kk` 和 `col`，`B[kk, col]` 会被同一输出 tile 中这一列的多个 `row` 使用；如果 tile 高度是 `TILE_M=16`，它理论上可被 16 个输出元素复用。
- 你写的 `2K * TILE * TILE -> 2K * TILE` 这个量级判断是对的，但它指的是 square output tile 下 `A/B` 的 global read 次数，不包含 `C` 写回，也默认把整个 `K` 方向按阶段搬进 shared memory 后复用。

更精确地说：对一个 `TILE_M x TILE_N` 输出 tile，naive 逻辑上每个输出元素读 `K` 个 `A` 和 `K` 个 `B`，所以 `A/B` 读约为 `2 * K * TILE_M * TILE_N` 个 float。理想 tiling 下，每个 `kk` 只需要为这个输出 tile 读 `TILE_M` 个 `A` 和 `TILE_N` 个 `B`，合计约 `K * (TILE_M + TILE_N)` 个 float。若 `TILE_M = TILE_N = TILE`，就是从 `2K * TILE * TILE` 降到 `2K * TILE`。

## v2 作业

### 题目 1
默认 `TILE=16` 时，推导一个 `16 x 16` 输出 tile 在 naive 和 v2 中对 `A/B` 的 global load 数量。为什么 `C` 的写回次数没有减少？

### 我的答案


### 自我检查


### 批改反馈


### 题目 2
下面这段 shared memory tile 代码有什么问题？至少指出两个 bug，并说明会导致死锁、越界还是结果错误。

```cpp
if (row >= m || col >= n) return;
for (int tile_k = 0; tile_k < k; tile_k += 16) {
    tile_a[ty][tx] = a[row * k + tile_k + tx];
    tile_b[ty][tx] = b[(tile_k + ty) * n + col];
    __syncthreads();
    for (int inner = 0; inner < 16; inner++) {
        sum += tile_a[ty][inner] * tile_b[inner][tx];
    }
}
```

### 我的答案


### 自我检查


### 批改反馈
