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

`TILE=16`时，v2情况下：每行/列仅从DRAM读一次，读TILE行/列，因此，A的global load数量`16 * K`, B的 global load数量`K * 16`；
naive情况下：若计算C的每个元素都要从DRAM读取A和B共`2 * K`，且因重复读取`TILE * TILE`次，总共要读取`2 * K * 16 * 16`，A和B的global load分别都是`K * 16 * 16`。

### 自我检查

忘了解释`C`的输出矩阵元素个数没变所以`C`的写回次数也不变。

### 批改反馈

结论：A/B 的数量级推导基本正确，但需要补上 `C` 写回为什么不变，并把“读一次”的范围说清楚。

关键要点：

- 对一个 `16 x 16` 输出 tile，naive 中每个 `C` 元素都要读 `K` 个 `A` 和 `K` 个 `B`。因此 `A` 的逻辑 global load 是 `16 * 16 * K = 256K` 个 float，`B` 也是 `256K` 个 float，合计 `512K` 个 float。
- v2 中这个输出 tile 会按 `K` 方向分阶段缓存。对完整 `K` 来说，`A` 只需要为这个 tile 读 `16 * K` 个 float，`B` 只需要读 `K * 16` 个 float，合计 `32K` 个 float。默认 `TILE=16` 时，`A/B` 的逻辑 global load 降为 naive 的 `1/16`。
- 你说的“每行/列仅从 DRAM 读一次”要限定在“同一个输出 tile 内、对每个 `kk`”这个范围。不同输出 tile 之间仍可能重复读同一段 `A/B`，实际是否来自 DRAM 还会受 L1/L2 cache 影响。
- `C` 的写回次数没有减少，因为输出矩阵的元素个数没有变。这个 `16 x 16` tile 仍然有 256 个 `C[row, col]`，每个最终结果都必须写回一次；tiling 只减少 `A/B` 的重复读取，不减少输出元素数量。

标准思路：先按“每个输出元素需要多少输入”算 naive，再按“一个 tile 每个 `kk` 只需要多少唯一的 `A/B` 元素”算 v2，最后单独统计 `C` 写回。

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

1、每轮sum累加结束都要__syncthreads进行block-level同步，不然可能会出现结果错误；
2、没有检查a和b数组访问是否越界，`tile_k + tx < k`和`tile_k + ty < k`；
3、这里显式指定了`kTile=16`，如果tx或者ty的blockDim不为16，会出现访问未覆盖完全的情况。

### 自我检查

新的点：在`__syncthreads()`之前提前`return`可能会组合成死锁风险。因为剩下的线程可能永远等不到`return`的线程。正确做法是越界线程给共享内存写`0.0f`。

在softmax和layernorm内没有错误是因为整个block线程一起退出了，因此不会出现与block其他线程同步问题。

### 批改反馈

结论：你指出了两个真正关键的问题：缺少 K 方向边界检查，以及缺少下一轮覆盖 shared memory 前的同步。还需要补充一个更严重的 bug：开头的提前 `return` 会和 `__syncthreads()` 组合成死锁风险。

关键要点：

- `if (row >= m || col >= n) return;` 放在 `__syncthreads()` 前是错误的。一个 block 里只要有部分线程越界提前返回，剩下线程进入 `__syncthreads()` 后就可能永远等不到这些线程，后果是死锁或 kernel hang。正确做法是所有线程都走到同步点，越界线程给 `tile_a/tile_b` 写 `0.0f`，最后写回 `C` 时再判断边界。
- 你写的 `tile_k + tx < k` 和 `tile_k + ty < k` 是对的。没有这两个检查时，如果 `K` 不是 16 的倍数，最后一个 `K` tile 会越界读取 `A` 或 `B`，结果可能错误，也可能触发非法访问。
- 第一条“每轮 sum 累加结束都要 `__syncthreads()`”方向正确。更精确地说，这个同步是为了防止有些线程已经进入下一轮 `tile_k` 并覆盖 `tile_a/tile_b`，而另一些线程还在读取当前轮的 shared memory；缺少它会造成 shared memory read/write race，通常表现为结果错误。
- 第 3 条在当前题设下不算主要 bug。这个版本本来就假设 `blockDim.x = blockDim.y = kTileK = 16`；如果以后要支持其他 block shape，确实需要重新设计 `tile_a/tile_b` 的维度和加载映射，但这不是这段代码相对 v2 正确写法的核心错误。

标准思路：这段代码至少要修三类问题：不能在 barrier 前让部分线程提前退出；每次 global load 都要做 `row/col/K` 边界检查并给 shared memory 补零；当前 tile 计算结束后要同步，再允许下一轮覆盖 shared memory。

## v3 作业

### 题目 1
v3 的输出 tile 是 `32 x 16`。请推导一个输出 tile 对 `A/B` 的 global load 数量，并说明为什么相对 v2 主要减少的是 `B` 读取而不是 `A` 读取。

### 我的答案


### 自我检查


### 批改反馈


### 题目 2
如果把 v3 改成 `1 x 2` register tile，一个线程计算同一行的两个相邻列，理论上会减少 `A` 还是 `B` 的有效 global load？它可能引入哪些新的代价？

### 我的答案


### 自我检查


### 批改反馈
