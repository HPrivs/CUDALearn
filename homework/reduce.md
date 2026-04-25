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

基本正确。你抓住了 v4 的核心：用 two-pass reduction 把 v3 中多个 block 对同一个 `y` 地址的全局 `atomicAdd`，改成第一轮写不同的 `partial[blockIdx.x]`，第二轮再归约 `partial`。

需要补一处更严谨的定量说法：v4 的 `AI` 没有明显提高，甚至按有效 DRAM 访存口径会略低一点，因为它额外写 `P` 个 partial、再读 `P` 个 partial，其中 `P = ceil(N / (blockDim.x * kItemsPerThread))`。默认 `N = 1 << 22`、`blockDim.x = 256`、`kItemsPerThread = 16` 时，`P = 1024`，额外 partial 读写只有约 `8 KB`，相对输入约 `16 MB` 很小。

标准答案可以压缩成一句：v4 没有减少输入读取和总加法次数，`AI` 基本不变，但它用一次额外 kernel launch 和很小的 partial 读写，换掉了 v3 剩余的同地址全局 `atomicAdd` 竞争，所以在默认大规模输入下可能更快。

### 题目 2
用一段话解释：为什么 reduce naive 版的 `AI` 比 vector add 高一点，但实际可能比 vector add 慢很多？

### 我的答案

reduce naive内核执行过程写入一次y，读取1次x[idx]，执行1次原子加法操作。访存量B为4N+4，操作量F为N - 1，AI=1/4 = 0.25,比vectorAdd的1/12要大。比vectorAdd慢是因为多个线程不能同时读写一个数据，不然会造成数据竞争，原子操作避免了竞争但必须等其他线程访问结束。

### 批改反馈

方向正确：`N` 很小时，v4 不一定比 v3 快，因为 two-pass 多出来的第二次 kernel launch 是固定开销，而小输入下能省掉的 atomic contention 很少。

需要修正一句术语：这里不应说“几百次的 `atomicAdd` 的 occupancy”。`occupancy` 指的是 SM 上活跃 warps/blocks 的占用情况，不是 atomic 次数或 atomic 开销。更准确的说法是：当 `N` 只有几千时，v3 的 `partial_count = ceil(N / (blockDim.x * kItemsPerThread))` 可能只有 1 个或很少几个 block，最后对 `y` 的全局 `atomicAdd` 次数很少，同地址竞争本来就不严重；这时 v4 再启动第二个 kernel 的固定开销可能超过去掉这些 atomic 的收益。

标准思路：

- 第二次 kernel launch 开销：v4 固定多启动一个 `kernel_v4_stage2`，小 `N` 时总计算量很少，这个固定成本难以摊薄。
- atomic contention：v3 的 atomic 次数等于 `partial_count`，小 `N` 下 `partial_count` 很小，竞争不明显；去掉它带来的收益有限。
- 结论：小规模下 v4 不保证更快，甚至可能慢于 v3；需要实测不同 `N`，观察 v3 和 v4 的交叉点。

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

借助register的快速访存特性，每个线程预先处理4个元素，进行树形规约的元素减少为了原来的四分之一，所需的线程也减少为了原来的四分之一，代表原子加法操作次数的grid数量同样变为了原先四分之一，虽然__syncthreads()和warp divergence的问题还在，但仍然减少了全局的`atomicAdd`次数。

### 自我检查

并不能说树形规约的元素减少为了原来的四分之一，因为输入元素总数仍然是`N`。 只是每个block需要处理的元素从`blockDim.x`变为了处理`blockDim.x * 4`, grid从`ceil(N / blockDim.x)` 变为了 `ceil(N / (blockDim.x * 4))`。

且register的快速访存特性不是核心收益，而是减少block reduce的局部结果数，block数和block合并次数。

### 批改反馈

基本正确，但有一个概念需要收紧：`kItemsPerThread = 4` 减少的是 block 数和每个 block 最后一次全局 `atomicAdd` 的次数，不是把“树形规约的元素减少为原来的四分之一”。输入元素总数仍然是 `N`，总加法次数仍约为 `N - 1`；只是每个线程先在 register 里串行累加 4 个元素，进入 shared memory 的局部和数量变成每 block `blockDim.x` 个。

所以更标准的表述是：每个 block 从处理 `blockDim.x` 个元素变成处理 `blockDim.x * 4` 个元素，grid 从 `ceil(N / blockDim.x)` 变成 `ceil(N / (blockDim.x * 4))`，因此全局 `atomicAdd` 次数约减少到 1/4。`AI` 不变，是因为有效 DRAM 访存仍然主要是读 `N` 个 `float` 加最终 1 个输出，FLOPs 仍然是求和所需的约 `N - 1` 次加法。

另外，“借助 register 的快速访存特性”这句话方向对，但核心收益不是 register 比 shared memory 快，而是 register 预累加减少了进入 block reduce 的局部结果数、block 数和跨 block 合并次数。

### 题目 2
预测题：如果把 `kItemsPerThread` 从 4 改成 16，性能一定会继续变好吗？请从 block 数、单线程串行工作、寄存器压力三个角度回答。


### 我的答案

kItemsPerThread=4:
v3_items4   ms:0.5055    GB/s: 33.19   TFLOPS: 0.0083   abs_err: 0.000366

kItemsPerThread=16:
v3_items16   ms:0.4610    GB/s: 36.39   TFLOPS: 0.0091   abs_err: 0.000366

从实测数据上看，性能有一些改善。但我认为继续增加kItemsPerThread参数性能不一定会继续变好。参数增大block数会减少相应的倍数，可能会限制的SM本身的性能。寄存器的角度看，每个线程寄存器的大小会分配16 * sizeof(float) Bytes的空间，有可能超出SM寄存器容量。单线程串行工作的角度看，参数增大会导致单线程的串行工作量增大，而GPU本身串行工作能力较弱，可能会限制性能。

### 自我检查

寄存器误区是并不会保存16个输入，而是可能会让编译器使用更多寄存器或更多指令，导致每线程寄存器数上升，occupancy下降。

### 批改反馈

判断正确：从 `4` 改到 `16` 不保证一定继续变好，你已经覆盖了题目要求的三个角度。

需要修正“每个线程寄存器的大小会分配 `16 * sizeof(float)`”不准确。当前 v3 写法里线程通常只需要一个 `sum` register 反复累加，不会因为循环 16 次就必然保存 16 个 float；真正可能增加的是循环变量、地址计算、展开后临时值和编译器调度带来的 register pressure。

更完整的标准思路：

- block 数：`K` 从 4 增到 16，会让 block 数和全局 `atomicAdd` 次数再降到约 1/4，这可能继续提升性能。
- 单线程串行工作：每个线程从最多累加 4 个元素变成最多 16 个元素，线程内并行度下降；如果全局 atomic 已不再是主要瓶颈，继续增加 `K` 的收益会变小，甚至被串行累加成本抵消。
- 寄存器压力：简单循环未必需要 16 个寄存器保存 16 个输入，但更大的 `K` 可能让编译器使用更多寄存器或更多指令；如果每线程寄存器数上升导致 occupancy 下降，性能可能变差。

你的实测里 `0.5055 ms -> 0.4610 ms` 说明在当前机器和当前规模下 `K = 16` 仍略快，但这只能支持“这次变快了”，不能推出“继续增大一定更快”。下一步如果要验证，应至少同时记录 `K = 1/4/8/16/32`，并保证输出版本名和实际 `K` 一致。

## v4 作业

### 题目 1
用一句话解释：为什么 v4 的 `AI` 没有明显提高，但仍然可能比 v3 更快？

### 我的答案

v4使用启动两次kernel的方法，首先第一个kernel在此前的基础上进行第一次规约访存partial，第二个kernel对partial规约得到的就是最终的规约值y，因为不需要跨块对同一地址进行大量atomicAdd。同时partial_count足够大，足以抵消两次启动内核的开销。

### 自我检查

v4的多了额外的partial读写，`2 * P * sizeof(float)`。总体来看，在FLOPS不变的情况下，`AI`甚至会比原先略低一些，但是相对于输入的规模变化不大，因此`AI`没有太大变化。

### 批改反馈

基本正确。你抓住了 v4 的核心：用 two-pass reduction 把 v3 中多个 block 对同一个 `y` 地址的全局 `atomicAdd`，改成第一轮写不同的 `partial[blockIdx.x]`，第二轮再归约 `partial`。

需要补一处更严谨的定量说法：v4 的 `AI` 没有明显提高，甚至按有效 DRAM 访存口径会略低一点，因为它额外写 `P` 个 partial、再读 `P` 个 partial，其中 `P = ceil(N / (blockDim.x * kItemsPerThread))`。默认 `N = 1 << 22`、`blockDim.x = 256`、`kItemsPerThread = 16` 时，`P = 1024`，额外 partial 读写只有约 `8 KB`，相对输入约 `16 MB` 很小。

标准答案可以压缩成一句：v4 没有减少输入读取和总加法次数，`AI` 基本不变，但它用一次额外 kernel launch 和很小的 partial 读写，换掉了 v3 剩余的同地址全局 `atomicAdd` 竞争，所以在默认大规模输入下可能更快。

### 题目 2
预测题：如果 `N` 很小，比如只有几千个元素，v4 还一定比 v3 快吗？请从第二次 kernel launch 开销和 atomic contention 两个角度回答。

### 我的答案

如果N只有几千个元素，partial_count的数量可能只有几百个，几百次的atomicAdd的occupancy很可能没有kernel launch的启动开销大。

### 自我检查

有点误区，`occupancy`指的是SM上活跃warps/blocks的占用情况，不指atomic的开销。同时从atomic contention和kernel launch的分析太笼统，直接看答案吧。


### 批改反馈

方向正确：`N` 很小时，v4 不一定比 v3 快，因为 two-pass 多出来的第二次 kernel launch 是固定开销，而小输入下能省掉的 atomic contention 很少。

需要修正一句术语：这里不应说“几百次的 `atomicAdd` 的 occupancy”。`occupancy` 指的是 SM 上活跃 warps/blocks 的占用情况，不是 atomic 次数或 atomic 开销。更准确的说法是：当 `N` 只有几千时，v3 的 `partial_count = ceil(N / (blockDim.x * kItemsPerThread))` 可能只有 1 个或很少几个 block，最后对 `y` 的全局 `atomicAdd` 次数很少，同地址竞争本来就不严重；这时 v4 再启动第二个 kernel 的固定开销可能超过去掉这些 atomic 的收益。

标准思路：

- 第二次 kernel launch 开销：v4 固定多启动一个 `kernel_v4_stage2`，小 `N` 时总计算量很少，这个固定成本难以摊薄。
- atomic contention：v3 的 atomic 次数等于 `partial_count`，小 `N` 下 `partial_count` 很小，竞争不明显；去掉它带来的收益有限。
- 结论：小规模下 v4 不保证更快，甚至可能慢于 v3；需要实测不同 `N`，观察 v3 和 v4 的交叉点。

## v5 作业

### 题目 1
用一句话解释：v5 的 `AI` 为什么和 v4 基本一样，但仍然可能略快？

### 我的答案

v5将v4的block reduce换成了warp reduce，降低了shared memory的访存流量，同时减少了block级的__syncthreads()，换为了更轻量的__shfl_down_sync();

### 自我检查


### 批改反馈

基本正确。你抓住了 v5 和 v4 的关键差异：v5 没有改变 two-pass 结构、输入读取次数或总加法次数，因此按有效 DRAM 访存计算的 `AI` 和 v4 基本一样；它可能略快，是因为 block 内归约从 shared memory tree 改成 warp-level shuffle，减少了 shared memory 读写和多轮 block-wide `__syncthreads()`。

需要把一句话再说严谨一点：v5 不是把整个 block reduce 都换成“纯 warp reduce”。它先在每个 warp 内用 `__shfl_down_sync` 归约，再把每个 warp 的 partial sum 写到 shared memory，最后由第一个 warp 继续归约这些 partial sums。所以 shared memory 没有完全消失，只是从 v4 的每个线程都写 `smem[tid]`、多轮读写，变成每个 warp 写 1 个结果，默认 `kBlockSize = 256` 时只写 8 个 `warp_sums`。

标准答案可以压缩成：v5 的 `B` 和 `F` 与 v4 基本相同，所以 `AI` 基本不变；但 v5 用 `__shfl_down_sync` 在 warp 内直接交换 register 值，减少 block 内 shared memory 访问和同步次数，因此在 block reduce 开销占比较高时可能略快。


### 题目 2
找 bug：如果把 `block_reduce_sum_v5` 里 `__syncthreads()` 删除，结果为什么可能不正确？

### 我的答案

需要等待所有warp把数据存入shared memory中，如果删去可能会出现读取未完成warp reduce的value的情况。

### 自我检查

“读取未完成warp reduce的value”说法不准确，应该说第一个warp可能读取到尚未写入或未定义的`warp_sums[i]`。`__shfl_down_sync`只能保证同一个warp内参与mask的lanes的同步交换register值。

### 批改反馈

正确。这个 `__syncthreads()` 的作用是保证所有 warp 的 lane 0 都已经把各自的 warp partial sum 写入 `warp_sums[warp_id]`，然后第一个 warp 才读取 `warp_sums[0..kWarpsPerBlock-1]` 做第二级归约。

更精确地说，问题不是“读取未完成 warp reduce 的 value”，而是第一个 warp 可能读取到尚未写入、旧值或未定义的 `warp_sums[i]`。`__shfl_down_sync` 只保证同一个 warp 内参与 mask 的 lanes 能同步交换 register 值；它不能同步不同 warp，也不能保证 shared memory 写入对其他 warp 可见。因此跨 warp 通过 shared memory 传递数据时，仍然需要 block-wide `__syncthreads()`。

标准思路：warp 内归约不需要 block 同步；warp 之间通过 shared memory 合并时需要同步。删掉这次同步后，结果会变成 race condition，可能偶尔对、偶尔错，取决于 warp 调度和 shared memory 写读先后。
