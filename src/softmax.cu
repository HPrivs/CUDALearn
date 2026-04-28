// Softmax: y[row, col] = exp(x[row, col] - max(row)) / sum_j exp(x[row, j] - max(row))
// I/O shape: x is M x K row-major, y is M x K row-major
// dtype: float32
// default problem size: M = 4096, K = 4096
// theoretical traffic per output element: read x 3 times + write y once = 16B
// reported FLOPs per output element: 2 subtracts + 1 add + 1 divide = 4 FLOPs
// note: comparisons and exp are not counted in reported FLOPs.

#include "common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>

namespace {

constexpr int kRows = 4096;
constexpr int kCols = 4096;
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

int div_up(int a, int b) {
    return (a + b - 1) / b;
}

void cpu_ref(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * cols;
        double max_val = static_cast<double>(x[row_offset]);
        for (int col = 1; col < cols; ++col) {
            max_val = std::max(max_val, static_cast<double>(x[row_offset + col]));
        }

        double denom = 0.0;
        for (int col = 0; col < cols; ++col) {
            denom += std::exp(static_cast<double>(x[row_offset + col]) - max_val);
        }

        for (int col = 0; col < cols; ++col) {
            double numer = std::exp(static_cast<double>(x[row_offset + col]) - max_val);
            y[row_offset + col] = static_cast<float>(numer / denom);
        }
    }
}

// ========= v1: naive multi-pass =========
// naive multi-pass 解决的问题：先用最直接的三次行扫描保证 softmax 数值稳定且正确。
__global__ void kernel_naive(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    
    const size_t row_offset = static_cast<size_t>(row) * cols;
    float max_val = -FLT_MAX;
    for (int col = 0; col < cols; col++) {
        max_val = fmaxf(max_val, x[row_offset + col]);
    }

    float denom = 0.0f;
    for (int col = 0; col < cols; col++) {
        denom += expf(x[row_offset + col] - max_val);
    }

    for (int col = 0; col < cols; col++) {
        y[row_offset + col] = expf(x[row_offset + col] - max_val) / denom;
    }
}

void launch_naive(const float* x, float* y, int rows, int cols) {
    int grid = div_up(rows, kBlockSize);
    kernel_naive<<<grid, kBlockSize>>>(x, y, rows, cols);
}

// ========= v2: block-per-row shared memory reduce =========
// block-per-row 解决的问题：让一个 block 内的多个线程并行处理同一行，减少行内串行等待。
__global__ void kernel_v2(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    const size_t row_offset = static_cast<size_t>(row) * cols;

    float max_val = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        max_val = fmaxf(max_val, x[row_offset + col]);
    }

    smem[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    max_val = smem[0];

    float denom = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        denom += expf(x[row_offset + col] - max_val);
    }
    smem[tid] = denom;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    denom = smem[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row_offset + col] = expf(x[row_offset + col] - max_val) / denom;
    }
}


void launch_v2(const float* x, float* y, int rows, int cols) {
    kernel_v2<<<rows, kBlockSize, kBlockSize * sizeof(float)>>>(x, y, rows, cols);
}

// ========= v3: warp shuffle block reduce =========
// warp shuffle 解决的问题：warp 内归约不再反复读写 shared memory，减少部分同步和片上访存。
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_vals[32];
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;
    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    val = warp_reduce_max(val);
    if (lane == 0) {
        warp_vals[warp_id] = val;
    }
    __syncthreads();

    val = (tid < num_warps) ? warp_vals[lane] : -FLT_MAX;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }
    if (tid == 0) {
        warp_vals[0] = val;
    }
    __syncthreads();
    return warp_vals[0];
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_vals[32];
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;
    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_vals[warp_id] = val;
    }
    __syncthreads();

    val = (tid < num_warps) ? warp_vals[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    if (tid == 0) {
        warp_vals[0] = val;
    }
    __syncthreads();
    return warp_vals[0];
}

__global__ void kernel_v3(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int tid = threadIdx.x;
    const size_t row_offset = static_cast<size_t>(row) * cols;

    float thread_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_max = fmaxf(thread_max, x[row_offset + col]);
    }
    float max_val = block_reduce_max(thread_max);

    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_sum += expf(x[row_offset + col] - max_val);
    }
    float denom = block_reduce_sum(thread_sum);

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row_offset + col] = expf(x[row_offset + col] - max_val) / denom;
    }
}

void launch_v3(const float* x, float* y, int rows, int cols) {
    kernel_v3<<<rows, kBlockSize>>>(x, y, rows, cols);
}

}  // namespace

int main() {
    const int rows = kRows;
    const int cols = kCols;
    const size_t elems = static_cast<size_t>(rows) * cols;
    const size_t bytes = elems * sizeof(float);
    const size_t traffic_bytes = elems * sizeof(float) * 4;
    const size_t flops = elems * 4;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    random_fill(d_x, elems, 2028);

    std::vector<float> h_x(elems), h_ref(elems), h_out(elems);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_x, h_ref, rows, cols);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    struct Version {
        const char* name;
        void (*launch)(const float*, float*, int, int);
    };

    const std::array<Version, 3> versions{{
        {"naive", launch_naive},
        {"v2_block", launch_v2},
        {"v3_warp", launch_v3},
    }};

    for (const auto& version : versions) {
        version.launch(d_x, d_y, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());
        if (err > 1e-5f) {
            std::cerr << "correctness failed: max_err=" << err << '\n';
            CUDA_CHECK(cudaFree(d_x));
            CUDA_CHECK(cudaFree(d_y));
            return 1;
        }

        float ms = timeit([&] { version.launch(d_x, d_y, rows, cols); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row(version.name, ms, traffic_bytes, flops, err);
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
