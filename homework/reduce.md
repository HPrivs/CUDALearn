# reduce 作业

## 使用说明
- 这个文件是 `reduce` 的答题纸。
- 你可以直接在每道题下面写答案，也可以插入 `cpp`、`cuda`、`bash` 代码块。
- 需要我批改时，直接说「批改 reduce 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，默写 `kernel_naive` 的核心逻辑：算 `idx`、边界检查、用 `atomicAdd` 累加到输出标量。

### 你的答案

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

### 题目 2
用一段话解释：为什么 reduce naive 版的 `AI` 比 vector add 高一点，但实际可能比 vector add 慢很多？

### 你的答案

reduce naive内核执行过程写入一次y，读取1次x[idx]，执行1次原子加法操作。访存量B为4N+4，操作量F为N - 1，AI=1/4 = 0.25,比vectorAdd的1/12要大。比vectorAdd慢是因为多个线程不能同时读写一个数据，不然会造成数据竞争，原子操作避免了竞争但必须等其他线程访问结束。

### 标准答案

reduce naive 每个元素大约做 1 次加法、读 1 个 float，总逻辑访存约 4N + 4 字节，所以 AI ≈ 0.25 FLOP/Byte，比 vector add 的 1/12 高。但它所有线程都对同一个输出地址执行 atomicAdd，同地址原子操作会发生严重竞争和串行化，很多线程在等待原子更新完成。因此实际性能可能远低于 vector add，瓶颈主要是 atomic contention，而不是普通的连续显存带宽。

## v2 作业

### 题目 1
用一句话解释：v2 的 `AI` 为什么几乎没变，但性能预期会明显变好？

### 你的答案


### 题目 2
找 bug：下面这段归约代码有什么问题？至少指出 1 个。

```cpp
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] += smem[tid + stride];
    }
}
```

### 你的答案
