# elementwise 作业

## 使用说明
- 这个文件是 `elementwise` 的答题纸。
- 你可以直接在每道题下面写答案，也可以插入 `cpp`、`cuda`、`bash` 代码块。
- 需要我批改时，直接说「批改 elementwise 作业」，我会默认读取这个文件。

## v1 作业

### 题目 1
合上代码，自己默写 `kernel_naive` 的核心三行：算 `idx`、做边界检查、完成 `c[idx] = a[idx] + b[idx]`。

### 我的答案

```cpp

__global__ void kernel_naive(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void launch_naive(const float* A, const float* B, float* C, int n) { 
    int grid = (n + kBlockSize - 1) / kBlockSize;
    kernel_naive<<<grid, kBlockSize>>>(A, B, C, n);
}
    
```

### 自我检查

### 批改反馈

### 题目 2
解释为什么这个算子的 `AI = 1/12` 很低，并用一句话判断它更像 memory-bound 还是 compute-bound。

### 我的答案

以我当前的显卡MX250为例，它的理论显存带宽约为48GB/s，这点与实验数据(3 * sizeof(float) * n float32 / ms得出的数据)相差无几。理论上的FP32 FLOPS是1.21 TFLOPS。

【理论上硬件平衡点是 `峰值算力 / 峰值带宽 = 1.21e12 / 48e9 = 25 FLOP / Byte` (MX 250为例)】

修改答案：vecAdd的加法次数是N次，读写数据的数据量是3 * size_float32 * N = 12 N。计算强度A=1/12 = 0.083 FLOP/Byte 远远小于25，且VecAdd每次处理一个元素只运算1次浮点加法，但却要读一次A[i],B[i]和写一次C[i]，更像是memory bound。

### 自我检查

### 批改反馈

## v2 作业

### 题目 1
用一段话回答：为什么 `float4` 版的 `B`、`F`、`AI` 都没变，但它仍然可能比 naive 更快？

### 我的答案

`float4`版没有改变总数据量和总计算量，所以`B`、`F`、`AI`都不变。因为naive版本每处理4个元素就要4次load(float A[i]), 4次load(float B[i]), 4次store(float C[i])；向量化访存版本每处理4个元素只需要1次load(float4 A[i]) 1次load(float4 B[i]), 1次store(float4 C[i])。访存指令数量之前的四分之一。另外地址计算和指令发射开销也减少了。

### 自我检查

### 批改反馈

### 题目 2
改错题：下面这段描述里有两个与向量化相关的问题，请指出来并解释为什么错。

- 直接把任意 `float*` 强转成 `float4*` 使用，但不检查尾部元素
- 看到 `float4` 后就断言这个 kernel 一定从 memory-bound 变成 compute-bound

### 我的答案

第一个问题，如果不能把所有float转化成float4,那么执行过程中必定出现未能覆盖剩余的float从而遗漏的情况。
实践中，我把tail部分代码删去，发现代码执行过程中并未出现错误，max_err没有检测出错误，我认为可能是kNum = 1 << 24的原因，所有float可转化为float4而无遗漏。将kNum修改为 kNum = (1 << 24) - 3 果然报错，first mismatch at idx=16777212, ref=-0.517136, got=0, abs_err=0.517136。未被覆盖部分的值仍是初始化的0。

第二个问题，memory-bound或者是compute-bound并不取决于数组是否使用了向量化访存，将float转化为float4并未改变总访存字节数或者总FLOPS，算法本身仍是memory-bound。判断瓶颈需要实际比较计算强度和硬件带宽/算力的平衡点。

### 自我检查

### 批改反馈

`float*` 不能无条件强转成 `float4*` 使用，至少要保证 `16B` 对齐，并且 N 不是 4 的倍数时要用标量路径处理尾部，否则最后 N % 4 个元素会漏算。

float4 也不能说明 kernel 变成 compute-bound，因为它只减少访存指令数和地址计算开销，没有减少总访存字节，也没有增加总 FLOPs；elementwise add 的 AI 仍是 1/12，通常仍是 memory-bound。


### 自己加的题目
默写向量化访问版本的`elementwise`。

### 我的答案

``` cpp
__global__ void kernel_float4(const float4* A, const float4* B, float4* C, int vec_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        float4 va = A[idx];
        float4 vb = B[idx];
        C[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}

void launch_kernel_float4(const float* A, const float* B, float* C, int n) {
    int vec_n = n / 4;
    int grid = (vec_n + kBlockSize - 1) / kBlockSize;

    // 此处应该判断vec_n != 0 (0 < n <= 3)以免grid = 0 【初版写错了】
    if (vec_n > 0) {
        kernel_float4<<<grid, kBlockSize>>>(
            reinterpret_cast<const float4*>(A), 
            reinterpret_cast<const float4*>(B),
            reinterpret_cast<float4*> (C), vec_n);
    }

    int tail = n - vec_n * 4;
    if (tail > 0) {

        kernel_naive<<<1, tail>>>(A + vec_n * 4, B + vec_n * 4, C + vec_n * 4, tail);
    }
}

``` 

### 自我检查

### 批改反馈
