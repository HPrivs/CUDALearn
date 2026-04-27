# transpose 作业

## 使用说明
- 这个文件是 `transpose` 的答题纸。
- 你可以直接在每道题下面写答案，也可以插入 `cpp`、`cuda`、`bash` 代码块。
- 需要我批改时，直接说「批改 transpose 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，自己默写 `kernel_naive` 的核心逻辑：计算 `row/col`、做二维边界检查、完成 `B[col * rows + row] = A[row * cols + col]`。

### 我的答案

```cpp

__global__ void kernel_naive(const float* a, float* b, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        b[static_cast<size_t>(col) * rows + row] = a[static_cast<size_t>(row) * cols + col];
    }
}

```


### 批改反馈

正确。`col` 映射到 `threadIdx.x`，`row` 映射到 `threadIdx.y`，二维边界检查也完整；输出下标 `col * rows + row` 和输入下标 `row * cols + col` 都对。使用 `static_cast<size_t>` 可以避免大矩阵线性下标用 `int` 溢出，这点很好。

### 题目 2
用一段话解释：为什么 naive transpose 的读侧通常较连续，但写侧通常不连续？

### 我的答案

连续的线程读取a的地址也是连续的，符合合并访存的条件。但对于b来说连续线程写入b是跨rows的并不连续。

### 批改反馈

相邻线程读 `A[row * cols + col]` 时，`col` 连续变化，所以读地址连续，容易形成 coalesced load；写 `B[col * rows + row]` 时，相邻线程的 `col` 连续变化但线性地址相差 `rows` 个 `float`，所以 store 是跨 stride 的，不容易合并成少量内存事务。

## v2 作业

### 题目 1
用自己的话解释：为什么 `v2` 写回 `B` 时要交换 `blockIdx.x` 和 `blockIdx.y`？如果不交换，会发生什么？

### 我的答案

v2的转置包含块内元素转置和块间转置。blockIdx.x和blockIdx.y交换意味着块位置交换到转置后的位置。如果不交换矩阵仅执行了块内元素转置。

### 自我检查

### 批改反馈

基本正确。你说的“块内元素转置 + 块间转置”抓住了 v2 的核心：`smem[threadIdx.x][threadIdx.y]` 负责 tile 内转置，交换 `blockIdx.x / blockIdx.y` 负责把整个 tile 放到转置后的矩阵位置。

补充一点：如果不交换 block 坐标，不只是“少了一步优化”，而是输出位置会错。每个 tile 会留在原来的 tile 坐标，只做了局部小块内部的转置；除非数据和 tile 分布刚好特殊，否则整体矩阵不是 `B[col, row] = A[row, col]`。

### 题目 2
改错题：下面代码有两个和 `v2` 相关的问题，请指出并说明后果。

```cpp
int out_col = blockIdx.x * kTileDim + threadIdx.x;
int out_row = blockIdx.y * kTileDim + threadIdx.y;

if (out_row < rows && out_col < cols) {
    b[out_row * cols + out_col] = smem[threadIdx.x][threadIdx.y];
}
```

### 我的答案

第一个错误：blockIdx.x与blockIdx.y没有进行交换，造成块位置错误。
第二个错误：out_row < cols && out_col < rows，转置后的矩阵的行列数应是交换的，这里的判断应该也要相应地交换。

### 自我检查

遗漏了下标也要做出修改，cols换成rows： `b[out_row * rows + out_col]`。

### 批改反馈

基本正确，但还漏了一个很关键的下标问题。

你指出的两个点都对：
1. `blockIdx.x / blockIdx.y` 没有交换，输出 tile 位置错误。
2. 输出矩阵 `B` 的形状是 `cols x rows`，所以边界应检查 `out_row < cols && out_col < rows`。

还需要补上：`B` 是 row-major，且每行长度是 `rows`，所以线性下标应是：

```cpp
b[static_cast<size_t>(out_row) * rows + out_col]
```

原代码写成 `out_row * cols + out_col`，在默认 `4096 x 4096` 方阵上会被掩盖；一旦 `rows != cols`，stride 就错了，轻则结果错误，重则可能访问到错误位置。

## v3 作业

### 题目 1
解释题：当前 `kTileDim = 32` 时，为什么 `v2` 的 `smem[threadIdx.x][threadIdx.y]` 读侧更容易出现严重 bank conflict？请用 `bank = addr % 32` 推导。

### 我的答案

smem[threadIdx.x][threadIdx.y]的地址addr = tid_x * 32 + tid_y; 
bank = (tid_x * 32 + tid_y) % 32 = tid_y。warp内线程正好访问同一个bank，造成bank conflict。

### 自我检查

### 批改反馈

基本正确。你写出了关键公式：

```text
addr = tid_x * 32 + tid_y
bank = (tid_x * 32 + tid_y) % 32 = tid_y
```

这说明在 `kTileDim = 32` 且一个 warp 通常对应固定 `tid_y`、连续 `tid_x = 0..31` 时，warp 内线程会落到同一个 bank 的不同地址，形成严重 bank conflict。

可以再补一句让答案更完整：`v3` padding 后行跨度变成 `33`，读 `smem[tid_x][tid_y]` 时 `bank = (tid_x * 33 + tid_y) % 32 = (tid_x + tid_y) % 32`，因此同一个 warp 会分散到 32 个 bank。

### 题目 2
扩展题：为什么 `v3` 的 padding 只改变 shared memory 行跨度，不需要修改 global memory 的 `A/B` 下标？

### 我的答案

因为kTileDim=32决定了块的大小，每次多出来的部分是重叠的，会被之后的transpose覆盖。

### 自我检查

思路错了，A和B的下标由矩阵定义决定，shared memory padding只影响smem的物理布局，代码只能访问到smem[tx][ty]中 `tx/ty < kTileDim的有效区域`。

### 批改反馈

这题目前不正确，主要问题是“多出来的部分是重叠的，会被之后的 transpose 覆盖”这个说法不对。

`v3` 的 padding 多出来的一列不是有效矩阵元素，也不会参与 global memory 写回；它只是让 shared memory 每一行的物理跨度从 `32` 个 `float` 变成 `33` 个 `float`。也就是说，padding 改的是片上暂存数组 `smem` 的地址映射，不改矩阵 `A/B` 的数学坐标。

标准思路可以写成：

```text
global memory 下标仍然由矩阵定义决定：
A[row, col] -> B[col, row]

shared memory padding 只影响 smem 的物理布局：
smem[32][32] -> smem[32][33]
```

所以 `A/B` 的下标、边界检查、输出矩阵形状都不需要因为 padding 改变；需要改变的只有 shared memory 的声明，让 `smem[tid_x][tid_y]` 的 bank 映射从 stride 32 变成 stride 33。
