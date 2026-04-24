# reduce 作业

## 使用说明
- 这个文件是 `reduce` 的答题纸。
- 你可以直接在每道题下面写答案，也可以插入 `cpp`、`cuda`、`bash` 代码块。
- 需要我批改时，直接说「批改 reduce 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，默写 `kernel_naive` 的核心逻辑：算 `idx`、边界检查、用 `atomicAdd` 累加到输出标量。

### 我的答案

``` cpp

__global__ void kernel_naive(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(y, x[idx]);
    }
}

void launch_naive(const float* x, float* y, int n) {
    CUDA_CHECK(cudaMemset(y, 0, sizeof(float)));
    int grid = (n + kBlockSize - 1) / kBlockSize;
    kernel_naive<<<grid, kBlockSize>>>(x, y, n);
}
```

### 批改反馈

### 题目 2
用一段话解释：为什么 reduce naive 版的 `AI` 比 vector add 高一点，但实际可能比 vector add 慢很多？

### 我的答案

reduce naive内核执行过程写入一次y，读取1次x[idx]，执行1次原子加法操作。访存量B为4N+4，操作量F为N - 1，AI=1/4 = 0.25,比vectorAdd的1/12要大。比vectorAdd慢是因为多个线程不能同时读写一个数据，不然会造成数据竞争，原子操作避免了竞争但必须等其他线程访问结束。

### 批改反馈

reduce naive 每个元素大约做 1 次加法、读 1 个 float，总逻辑访存约 4N + 4 字节，所以 AI ≈ 0.25 FLOP/Byte，比 vector add 的 1/12 高。但它所有线程都对同一个输出地址执行 atomicAdd，同地址原子操作会发生严重竞争和串行化，很多线程在等待原子更新完成。因此实际性能可能远低于 vector add，瓶颈主要是 atomic contention，而不是普通的连续显存带宽。

## v2 作业

### 题目 1
用一句话解释：v2 的 `AI` 为什么几乎没变，但性能预期会明显变好？

### 我的答案

相比与v1, v2的访存量和FLOPS只改变了共享内存的部分。v2引入了块内共享内存，将数据存入共享内存并使用树形规约将规约存于每块smem[0]位置，原子加法操作减少为了原先的1/kBlockSize。

### 自我检查

需要了解到：shared memory的访存和FLOPS不计入DRAM访存（全局访存）`B`中，因为shared memory属于片上的访存。

### 批改反馈

基本正确。关键点抓到了：v2 的有效全局访存仍主要是读 `N` 个 `float` 和写 1 个输出，FLOPs 仍约为 `N - 1`，所以按有效访存算出来的 `AI` 几乎不变。性能预期变好不是因为 `AI` 提高，而是因为全局同地址 `atomicAdd` 从每个元素一次降到每个 block 一次，默认 `kBlockSize = 256` 时大约减少 256 倍。

需要稍微修正一句：shared memory 的读写会增加片上访存和同步开销，但通常不计入这里的有效 DRAM 访存 `B`。所以更严谨的说法是：算法级有效访存和 FLOPs 基本不变，但全局原子竞争大幅减少。


### 题目 2
找 bug：下面这段归约代码有什么问题？至少指出 1 个。

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] += smem[tid + stride];
    }
}
```

### 我的答案

缺少了等待线程同步__syncthreads()，这会导致未完成树形规约操作的情况下，提前将smem[0]累积到y中。

### 自我检查

没有触及到核心，每次循环如果不等待轮与轮之间线程同步，可能会造成某些线程读取上一轮还未完成写入的smem值。

### 批改反馈

正确。这个循环每一轮 `smem[tid] += smem[tid + stride]` 之后都需要 `__syncthreads()`，否则下一轮 stride 更小时，某些线程可能读到上一轮还没写完的 `smem` 值，导致归约结果不确定。

你的解释里“提前将 `smem[0]` 累积到 `y`”是最终表现之一，但更核心的问题发生在 loop 内部：轮与轮之间缺少同步，会产生 shared memory 的读写顺序问题。如果代码片段还包含加载 `smem[tid]` 的前文，那么开始归约前也需要一次 `__syncthreads()`。

## v3 作业

### 题目 1
用一句话解释：为什么 `kItemsPerThread = 4` 可以减少全局 `atomicAdd` 次数，但不改变 reduce 的 `AI`？

### 我的答案

### 自我检查

### 批改反馈

### 题目 2
预测题：如果把 `kItemsPerThread` 从 4 改成 16，性能一定会继续变好吗？请从 block 数、单线程串行工作、寄存器压力三个角度回答。

### 我的答案

### 自我检查

### 批改反馈
