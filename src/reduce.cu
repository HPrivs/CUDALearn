// Reduce Sum: y = sum_i x[i]
// I/O shape: x is a 1D vector with N elements, y is one scalar
// dtype: float32
// default problem size: N = 1 << 24
// theoretical traffic per element: read x[i] 4B, plus one final output float
// theoretical FLOPs per element: approximately 1 floating-point add

#include "common.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr int kNumElems = (1 << 24);
constexpr int kBenchmarkWarmup = 3;
constexpr int kBenchmarkIters = 20;

template <typename Launcher>
void benchmark_version(const char* version, Launcher&& launcher, const float* d_x,
                       float* d_y, float* d_partial, int n, size_t traffic_bytes,
                       size_t flops, const std::vector<float>& h_ref,
                       std::vector<float>& h_out, int warmup, int iters) {
    launcher(d_x, d_y, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, sizeof(float), cudaMemcpyDeviceToHost));
    float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());

    float ms = timeit([&] { launcher(d_x, d_y, d_partial, n); }, warmup, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_row(version, ms, traffic_bytes, flops, err);
}


int div_up(int a, int b) {
    return (a + b - 1) / b;
}

void cpu_ref(const std::vector<float>& x, std::vector<float>& y) {
    double sum = 0.0;
    for (float v : x) {
        sum += static_cast<double>(v);
    }
    y[0] = static_cast<float>(sum);
}


// ========= v1: naive atomicAdd =========
// 为什么这么做：先用最直接的全局 atomicAdd 保证正确性，后面再逐步减少原子操作和同步开销。
__global__ void kernel_naive(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(y, x[idx]);
    }
}

void launch_naive(const float* x, float* y, float* /*partial*/, int n) {
    CUDA_CHECK(cudaMemset(y, 0, sizeof(float)));
    int grid = div_up(n, kBlockSize);
    kernel_naive<<<grid, kBlockSize>>>(x, y, n);
}

// ========= v2: shared memory block reduce =========
// shared memory 是 block 内线程共享的片上存储；先在 block 内求局部和，可以把全局 atomicAdd 次数大幅减少。
__global__ void kernel_v2(const float* x, float* y, int n) {

    // block内线程共享，线程块大小
    __shared__ float smem[kBlockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < n) ? x[idx] : 0.0f;

    __syncthreads();

    // stride初始值为线程块的一半，每次减半(stride右移一位)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // 每个线程块的smem[0]存储最终的规约结果，使用此结果进行原子加法
    if (tid == 0) {
        atomicAdd(y, smem[0]);
    }
}

void launch_v2(const float* x, float* y, float* /*partial*/, int n) {
    CUDA_CHECK(cudaMemset(y, 0, sizeof(float)));
    int grid = div_up(n, kBlockSize);
    kernel_v2<<<grid, kBlockSize>>>(x, y, n);
}

// ========= v3: per-thread accumulation =========
// 每个线程先在寄存器里累加多个连续批次的元素，再进入 shared memory 归约；
// 这样可以减少 block 数和最终全局 atomicAdd 次数。
__global__ void kernel_v3(const float* x, float* y, int n) {
    __shared__ float smem[kBlockSize]; 

    int tid = threadIdx.x;
    
    int block_start = blockIdx.x * blockDim.x * kItemsPerThread;
    float sum = 0.0f;

    // 此处仍然符合coalescing，合并访存看warp内所有线程访问的地址是否连续，
    // 不是看单个线程前后两次访问是否连续
    for (int i = 0; i < kItemsPerThread; ++i) {
        int idx = block_start + i * blockDim.x + tid;
        if (idx < n) {
            sum += x[idx];
        }
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(y, smem[0]);
    }
}

void launch_v3(const float* x, float* y, float* /*partial*/, int n) {
    CUDA_CHECK(cudaMemset(y, 0, sizeof(float)));
    int elems_per_block = kBlockSize * kItemsPerThread;

    // n不变，每个线程块处理 kItemsPerThread 倍元素，grid 相应缩小。
    int grid = div_up(n, elems_per_block);
    kernel_v3<<<grid, kBlockSize>>>(x, y, n);

}

// ========= v4: two-pass reduction =========
// two-pass reduction 先把每个 block 的局部和写到 partial 数组，再二次归约，避免所有 block 争抢同一个 y。
__global__ void kernel_v4_stage1(const float* x, float* partial, int n) {
    __shared__ float smem[kBlockSize];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * kItemsPerThread;
    float sum = 0.0f;

    for (int i = 0; i < kItemsPerThread; ++i) {
        int idx = block_start + i * blockDim.x + tid;
        if (idx < n) {
            sum += x[idx];
        }
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}


__global__ void kernel_v4_stage2(const float* partial, float* y, int partial_count) {
    __shared__ float smem[kBlockSize];

    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int idx = tid; idx < partial_count; idx += blockDim.x) {
        sum += partial[idx];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[0] = smem[0];
    }
}

void launch_v4(const float* x, float* y, float* partial, int n) {
    int partial_count = div_up(n, kBlockSize * kItemsPerThread);
    kernel_v4_stage1<<<partial_count, kBlockSize>>>(x, partial, n);
    kernel_v4_stage2<<<1, kBlockSize>>>(partial, y, partial_count);
}

// ========= v5: warp shuffle block reduce =========
// warp shuffle 让同一个 warp 内的线程直接交换 register 值，
// 减少 shared memory 读写和部分 __syncthreads()。
__device__ float warp_reduce_sum(float value) {
    // warp内规约
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        // 0xffffffff是warp掩膜; value表示给其他线程读取的register值; delta=offset向下读取距离
        // __shfl_down_sync本身是warp-level同步交换指令，不用担心其他lane不同步
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_reduce_sum_v5(float value, float* warp_sums) {
    int tid = threadIdx.x;
    // 线程在warp内的编号，kWarpSize-1作掩码，范围在0~31
    int lane = tid & (kWarpSize - 1);
    // 当前warp的编号 
    int warp_id = tid / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }

    // 等待其他warp将value写入shared_memory
    __syncthreads();

    // tid = (0 ~ 7) 的register置为warp_sums[tid]，其他线程置为0，防止归并时数据污染。
    value = (tid < kWarpsPerBlock) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

__global__ void kernel_v5_stage1(const float* x, float* partial, int n) {
    __shared__ float warp_sums[kWarpsPerBlock];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * kItemsPerThread;

    float sum = 0.0f;
    for (int i = 0; i < kItemsPerThread; ++i) {
        int idx = block_start + i * blockDim.x + tid;
        if (idx < n) {
            sum += x[idx];
        }
    }

    sum = block_reduce_sum_v5(sum, warp_sums);
    if (tid == 0) {
        partial[blockIdx.x] = sum;
    }
}

__global__ void kernel_v5_stage2(const float* partial, float* y, int partial_count) {
    __shared__ float warp_sums[kWarpsPerBlock];

    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int idx = tid; idx < partial_count; idx += blockDim.x) {
        sum += partial[idx];
    }

    sum = block_reduce_sum_v5(sum, warp_sums);
    if (tid == 0) {
        y[0] = sum;
    }
}

void launch_v5(const float* x, float* y, float* partial, int n) {
    int partial_count = div_up(n, kBlockSize * kItemsPerThread);
    kernel_v5_stage1<<<partial_count, kBlockSize>>>(x, partial, n);
    kernel_v5_stage2<<<1, kBlockSize>>>(partial, y, partial_count);
}



}  // namespace

int main() {
    const int n = kNumElems;
    const size_t input_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t output_bytes = sizeof(float);
    const size_t traffic_bytes = input_bytes + output_bytes;
    const int partial_count = div_up(n, kBlockSize * kItemsPerThread);
    const size_t partial_bytes = static_cast<size_t>(partial_count) * sizeof(float);
    const size_t traffic_bytes_v4 = input_bytes + 2 * partial_bytes + output_bytes;
    const size_t flops = static_cast<size_t>(n - 1);


    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, partial_bytes));

    random_fill(d_x, n, 2028);

    std::vector<float> h_x(n), h_ref(1), h_out(1);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, input_bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_x, h_ref);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    benchmark_version("naive", launch_naive, d_x, d_y, d_partial, n, traffic_bytes,
                      flops, h_ref, h_out, kBenchmarkWarmup, kBenchmarkIters);
    benchmark_version("v2_smem", launch_v2, d_x, d_y, d_partial, n, traffic_bytes,
                      flops, h_ref, h_out, kBenchmarkWarmup, kBenchmarkIters);
    benchmark_version("v3_items16", launch_v3, d_x, d_y, d_partial, n, traffic_bytes,
                      flops, h_ref, h_out, kBenchmarkWarmup, kBenchmarkIters);
    benchmark_version("v4_two_pass", launch_v4, d_x, d_y, d_partial, n,
                      traffic_bytes_v4, flops, h_ref, h_out, kBenchmarkWarmup,
                      kBenchmarkIters);
    benchmark_version("v5_warp_shuffle", launch_v5, d_x, d_y, d_partial, n,
                      traffic_bytes_v4, flops, h_ref, h_out, kBenchmarkWarmup,
                      kBenchmarkIters);

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
