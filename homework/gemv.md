# gemv 作业

## 使用说明
- 这个文件是 `gemv` 的答题纸。
- 你可以直接在每道题下面写答案，也可以插入 `cpp`、`cuda`、`bash` 代码块。
- 需要我批改时，直接说「批改 gemv 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，默写 `kernel_naive` 的核心逻辑：计算 `row`、边界检查、循环 `col` 完成 dot product、写回 `y[row]`。

### 我的答案

``` cpp

__global__ void kernel_naive(const float* a, const float* x, float* y, 
                            int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        size_t row_offset = static_cast<size_t>(row) * cols;
        for (int col = 0; col < cols; col++) {
            sum += a[row_offset + col] * x[col];
        }
        y[row] = sum;
    }
}

```

### 自我检查


### 批改反馈

正确。现在这版已经覆盖了 naive kernel 的核心逻辑：

- `row = blockIdx.x * blockDim.x + threadIdx.x`：一维 grid/block 映射到输出行。
- `if (row < rows)`：处理 `rows` 不能被 `blockDim.x` 整除的边界。
- `row_offset = row * cols`：定位当前行在 row-major 矩阵里的起点。
- 循环 `col`，累加 `a[row_offset + col] * x[col]`：符合 `y[row] = sum(A[row, col] * x[col])`。
- 最后写回 `y[row]`：每个线程只写一个输出元素，没有写冲突。

小提醒：项目里的 GPU naive kernel 用 `float sum = 0.0f`，CPU reference 才用 `double` 做中间累加。你这里写 `double sum` 在数学上没问题，但会改变 GPU 上的指令类型和性能特征；如果目标是默写当前 `kernel_naive`，建议写成 `float sum = 0.0f`。



### 题目 2
用一段话解释：为什么 GEMV naive 版的 `AI` 约为 `0.25 FLOP/Byte`，但实际 DRAM 访存可能小于文档里的逻辑 `B`？

### 我的答案

硬件可能会把向量缓存到l1或者l2 cache中，这样就不会重复从DRAM中读取向量x，这样实际DRAM访存就会小于`B`。

### 自我检查


### 批改反馈

正确。关键点抓住了：文档里的 `B = M*K*4 + M*K*4 + M*4` 是按“每一行都逻辑读取一次整段 `x`”计算的有效访存字节；但真实硬件访问 DRAM 时，`x` 可能被 L2 cache 或 L1 cache 复用，所以不一定每次都重新从 DRAM 取。

可以再补一句让答案更完整：`A` 的每一行基本只读一次，复用机会少；`x` 被所有行反复读取，才是 cache 能让实际 DRAM bytes 小于逻辑 `B` 的主要来源。因此 naive 版的 `AI ≈ 0.25 FLOP/Byte` 是逻辑口径下的分析值，不等价于 profiler 里看到的真实 DRAM 字节口径。

## v2 作业

### 题目 1
概念题：为什么 `v2` 的 `AI` 和 naive 几乎相同，但实测速度能明显变快？请用“每行协作线程数”和“每线程串行循环长度”解释。

### 我的答案

v1内核让线程块处理多行A，而每行分配的协作线程数为1,导致了每个线程所需处理K列的数据，因此每线程串行循环长度为K。v2内核则让一个线程块处理一行A，这样每行协作线程数则为kBlockSize，每线程串行循环长度为div_up(K, kBlockSize)，行内并行数从1提升到了kBlockSize，因此速度明显变快。

### 自我检查


### 批改反馈

正确。

关键理解是对的：`v1/naive` 中一个线程独自处理一整行，所以每个线程要串行循环 `K` 次；`v2` 中一个 block 处理一行，默认 `kBlockSize = 256` 个线程协作，因此每个线程大约只处理 `ceil(K / 256)` 个列元素。`AI` 仍然接近 `0.25 FLOP/Byte`，因为有效读写字节和有效 FLOPs 没有变；速度提升主要来自行内并行度提升，而不是算法的算术强度提升。


所以这题的标准表达是：`v2` 把每行协作线程数从 `1` 提升到 `kBlockSize`，把每个线程的串行循环长度从 `K` 降到约 `ceil(K / kBlockSize)`，因此即使 `AI` 几乎不变，实测也能明显变快。


### 题目 2
改错题：下面代码少了两个关键点，请指出问题和可能后果。

```cpp
__global__ void kernel_v2_bug(const float* a, const float* x, float* y,
                              int rows, int cols) {
    __shared__ float smem[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        sum += a[row * cols + col] * x[col];
    }
    smem[tid] = sum;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
    }
    if (tid == 0) {
        y[row] = smem[0];
    }
}
```

### 我的答案

1、没有判断row < rows，可能发生y[row], a[row * cols + col]数组越界。
2、同时每次为smem赋值并没有__syncthreads()同步线程块，这样可能导致smem规约数据混乱，y[row]也相应出错。


### 自我检查

2应该说保证smem读取的是上一轮规约后的结果，防止读取旧值或者未写完值。

### 批改反馈

正确，两个关键 bug 都指出来了。

1. 缺少 `row < rows` 边界检查：如果 launch 的 grid 大于 `rows`，会访问 `a[row * cols + col]` 和 `y[row]` 的越界地址。当前项目里 `launch_v2` 用 `kernel_v2<<<rows, kBlockSize>>>`，刚好不会产生多余 block，但健壮写法仍应保留边界判断，尤其是以后改 grid 组织方式时。
2. 缺少 `__syncthreads()`：`smem[tid] = sum` 后必须同步，保证所有线程都写完 partial sum；每一轮 `smem[tid] += smem[tid + stride]` 后也必须同步，保证下一轮读取的是上一轮归约后的结果。否则会出现读旧值、读未写完值，最终 `y[row]` 不稳定或错误。

更完整的标准思路是：写入 `smem` 后一次同步，归约循环中每轮加法后一次同步，最后只让 `tid == 0` 写回结果；如果 grid 可能超过 `rows`，读写 `a/y` 都要受 `row < rows` 保护。

## v3 作业

### 题目 1
预测题：如果把 `kRowsPerBlock` 从 `4` 改成 `8`，每行协作线程数会变成多少？它可能更快还是更慢？请从“每行线程数”和“block 数”两个角度说明。

### 我的答案

改成8每行协作线程变成`ceil(kBlockSize / 8) = 32`，此时每个线程需要处理的该行数量为`4096 / 32 = 128`，线程串行循环次数从`64`提升到了`128`，block数从`ceil(rows, kRowsPerBlock) = 1024`变为了`512`。总体上看它减少了block数的同时增加了线程串行循环次数和规约次数且减少了并行度，当K规模大时，GPU可能不够吃满并行带宽，这种情况下可能是更慢的。

实测下：
每块处理4行时，ms和GB/s：1.4256       94.16
8行时：1.4392       93.27

可以看到是整体是略微变慢。

### 自我检查

弄混了一点，每行协作线程数减少是减少每行规约次数的，需要规约的数据量`64 -> 32`减少了。

### 批改反馈

基本正确，预测和实测结论都合理：`kRowsPerBlock = 8` 时，每个 block 仍是 `256` 个线程，所以每行协作线程数变成 `256 / 8 = 32`。默认 `K = 4096` 下，每个线程大约处理 `4096 / 32 = 128` 个元素；block 数从 `ceil(4096 / 4) = 1024` 降到 `ceil(4096 / 8) = 512`。

需要修正一点：这里不是“规约次数增加”。每行线程数从 `64` 降到 `32` 后，shared memory 归约轮数从 `log2(64) = 6` 降到 `log2(32) = 5`，所以每行归约轮数反而减少。真正增加的是每个线程的串行循环长度，也就是从约 `64` 个元素增加到约 `128` 个元素。

你的实测 `1.4256 ms -> 1.4392 ms` 说明在当前机器和默认规模下，`kRowsPerBlock = 8` 略慢。一个合理解释是：block 数继续减少带来的收益已经不明显，而每行只有 32 个线程后，行内访存并行度下降、单线程循环更长，抵消甚至超过了更少 block 和更少归约轮数的收益。

标准思路可以压缩成一句：`kRowsPerBlock = 8` 会把每行线程数降到 `32`、block 数降到 `512`；它可能因调度和归约开销更少而更快，也可能因行内并行度不足、每线程串行循环更长而更慢，当前实测是略慢。


### 题目 2
概念题：为什么 `v3` 的 `AI` 和 `v2` 几乎相同，但 `v3` 仍可能更快？

### 我的答案

v3的没有改变DRAM访存字节数和TFLOPS，但v3由每块处理一行改为了每块处理四行，虽然减少了每行并行数，但是减少了规约次数和block数，在K规模一般的情况下，节约的调度成本和规约粒度，超过了访存并行度减少带来的负面影响。

### 自我检查

没有改变的是有效FLOPS，`TFLOPS`是实测吞吐率。

### 批改反馈

正确。`v3` 和 `v2` 的有效 `AI` 几乎相同，因为它们计算的是同一个 GEMV：有效 FLOPs 仍是 `M * (2K - 1)`，有效访存字节仍按读 `A`、逻辑读 `x`、写 `y` 统计。

需要把一句话改得更精确：不是“不改变 TFLOPS”，而是“不改变有效 FLOPs”。`TFLOPS` 是实测吞吐率，会随着运行时间变化；这次 `v3` 更快，所以实测 `TFLOPS` 也略高。

你抓住了核心：`v3` 可能更快不是因为 `AI` 提高，而是因为 block 组织改变了。它把每个 block 处理的行数从 1 行变成 4 行，减少 block 数，并把每行归约轮数从 8 轮降到 6 轮；只要每行 64 个线程仍足够提供访存并行度，这些调度和归约开销的节省就可能超过每线程循环变长的代价。
