// LayerNorm: y[row, col] = (x[row, col] - mean(row)) / sqrt(var(row) + eps)
// mean(row) = sum_j x[row, j] / K
// var(row) = sum_j (x[row, j] - mean(row))^2 / K
// RMSNorm: y[row, col] = x[row, col] / sqrt(mean_square(row) + eps)
// mean_square(row) = sum_j x[row, j]^2 / K
// I/O shape: x is M x K row-major, y is M x K row-major
// dtype: float32
// default problem size: M = 4096, K = 4096
// theoretical traffic per output element:
//   v1/v2: read x 3 times + write y once = 16B
//   v3/v4: read x 2 times + write y once = 12B
// reported FLOPs per output element:
//   v1/v2: 6 FLOPs; v3: 7 FLOPs; v4: 3 FLOPs.
//   sqrt/div and row-level scale ops are not counted.

#include "common.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kRows = 4096;
constexpr int kCols = 4096;
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr float kEps = 1e-5f;

int div_up(int a, int b) {
    return (a + b - 1) / b;
}


void cpu_ref(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        const size_t row_offset = static_cast<size_t>(row) * cols;

        double sum = 0.0;
        for (int col = 0; col < cols; col++) {
            sum += static_cast<double>(x[row_offset + col]);
        }
        const double mean = sum / static_cast<double>(cols);

        double sq_sum = 0.0;
        for (int col = 0; col < cols; col++) {
            double diff = static_cast<double>(x[row_offset + col]) - mean;
            sq_sum += diff * diff;
        }

        const double var = sq_sum / static_cast<double>(cols);
        const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(kEps));

        for (int col = 0; col < cols; col++) {
            y[row_offset + col] =
                static_cast<float>((static_cast<double>(x[row_offset + col]) - mean) * inv_std);
        }
    }
}

void cpu_ref_rmsnorm(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        const size_t row_offset = static_cast<size_t>(row) * cols;

        double sq_sum = 0.0;
        for (int col = 0; col < cols; col++) {
            const double val = static_cast<double>(x[row_offset + col]);
            sq_sum += val * val;
        }

        const double mean_square = sq_sum / static_cast<double>(cols);
        const double inv_rms = 1.0 / std::sqrt(mean_square + static_cast<double>(kEps));

        for (int col = 0; col < cols; col++) {
            y[row_offset + col] =
                static_cast<float>(static_cast<double>(x[row_offset + col]) * inv_rms);
        }
    }
}

// ========= v1: naive multi-pass =========
// naive multi-pass 解决的问题：先用最直接的三次行扫描保证 LayerNorm 公式和边界处理正确。
__global__ void kernel_naive(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const size_t row_offset = static_cast<size_t>(row) * cols;

    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += x[row_offset + col];
    }
    float mean = sum / static_cast<float>(cols);

    float sq_sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        const float diff = x[row_offset + col] - mean;
        sq_sum += diff * diff;
    }

    const float var = sq_sum / static_cast<float>(cols);
    const float inv_std = 1.0f / sqrtf(var + kEps);

    for (int col = 0; col < cols; col++) {
        y[row_offset + col] = (x[row_offset + col] - mean) * inv_std;
    }

}


void launch_naive(const float* x, float* y, int rows, int cols) {
    const int grid = div_up(rows, kBlockSize);
    kernel_naive<<<grid, kBlockSize>>>(x, y, rows, cols);
}

// ========= v2: block-per-row + warp shuffle reduce =========
// block-per-row 解决的问题：用一个 block 并行处理一行，避免单线程串行扫完整行。
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_vals[32];

    int tid = threadIdx.x;
    int lane = tid & (kWarpSize - 1);
    int warp_id = tid / kWarpSize;
    int num_warps = blockDim.x / kWarpSize;

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


__global__ void kernel_v2(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int tid = threadIdx.x;

    const size_t row_offset = static_cast<size_t>(row) * cols;
    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_sum += x[row_offset + col];
    }
    const float sum = block_reduce_sum(thread_sum);
    const float mean = sum / static_cast<float>(cols);

    float thread_sq_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float diff = x[row_offset + col] - mean;
        thread_sq_sum += diff * diff;
    }
    const float sq_sum = block_reduce_sum(thread_sq_sum);
    const float var = sq_sum / static_cast<float>(cols);
    const float inv_std = 1.0f / sqrtf(var + kEps);

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row_offset + col] = (x[row_offset + col] - mean) * inv_std;
    }
}

void launch_v2(const float* x, float* y, int rows, int cols) {
    kernel_v2<<<rows, kBlockSize>>>(x, y, rows, cols);
}

// ========= v3: Welford variance =========
// Welford 解决的问题：在一次扫描中稳定地合并 mean 和 variance 统计量，减少一次 global read。
struct WelfordData {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ WelfordData make_welford_data() {
    return {0.0f, 0.0f, 0};
}

__device__ __forceinline__ WelfordData welford_update(WelfordData acc, float x) {
    acc.count += 1;
    const float delta = x - acc.mean;
    acc.mean += delta / static_cast<float>(acc.count);
    const float delta2 = x - acc.mean;
    acc.m2 += delta * delta2;
    return acc;
}

__device__ __forceinline__ WelfordData welford_combine(WelfordData a, WelfordData b) {
    if (a.count == 0) {
        return b;
    }
    if (b.count == 0) {
        return a;
    }

    const int count = a.count + b.count;
    const float count_f = static_cast<float>(count);
    const float a_count = static_cast<float>(a.count);
    const float b_count = static_cast<float>(b.count);
    const float delta = b.mean - a.mean;

    WelfordData out;
    out.count = count;
    out.mean = a.mean + delta * (b_count / count_f);
    out.m2 = a.m2 + b.m2 + delta * delta * (a_count * b_count / count_f);
    return out;
}

__device__ __forceinline__ WelfordData warp_reduce_welford(WelfordData val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        WelfordData other;
        other.mean = __shfl_down_sync(0xffffffff, val.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, val.count, offset);
        val = welford_combine(val, other);
    }
    return val;
}

__device__ __forceinline__ WelfordData block_reduce_welford(WelfordData val) {
    __shared__ WelfordData warp_vals[32];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    val = warp_reduce_welford(val);
    if (lane == 0) {
        warp_vals[warp_id] = val;
    }
    __syncthreads();

    val = (tid < num_warps) ? warp_vals[lane] : make_welford_data();
    if (warp_id == 0) {
        val = warp_reduce_welford(val);
    }
    if (tid == 0) {
        warp_vals[0] = val;
    }
    __syncthreads();
    return warp_vals[0];
}

__global__ void kernel_v3(const float* x, float* y, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const size_t row_offset = static_cast<size_t>(row) * cols;

    WelfordData thread_stat = make_welford_data();
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_stat = welford_update(thread_stat, x[row_offset + col]);
    }
    const WelfordData row_stat = block_reduce_welford(thread_stat);
    const float mean = row_stat.mean;
    const float var = row_stat.m2 / static_cast<float>(cols);
    const float inv_std = 1.0f / sqrtf(var + kEps);

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row_offset + col] = (x[row_offset + col] - mean) * inv_std;
    }
}

void launch_v3(const float* x, float* y, int rows, int cols) {
    kernel_v3<<<rows, kBlockSize>>>(x, y, rows, cols);
}

// ========= v4: RMSNorm =========
// RMSNorm 解决的问题：去掉 mean 统计，只用均方根归一化，降低统计阶段和归约对象复杂度。
__global__ void kernel_v4(const float* x, float* y, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const size_t row_offset = static_cast<size_t>(row) * cols;

    float thread_sq_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float val = x[row_offset + col];
        thread_sq_sum += val * val;
    }

    const float sq_sum = block_reduce_sum(thread_sq_sum);
    const float mean_square = sq_sum / static_cast<float>(cols);
    const float inv_rms = 1.0f / sqrtf(mean_square + kEps);

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row_offset + col] = x[row_offset + col] * inv_rms;
    }
}


void launch_v4(const float* x, float* y, int rows, int cols) {
    kernel_v4<<<rows, kBlockSize>>>(x, y, rows, cols);
}

}  // namespace

int main() {
    const int rows = kRows;
    const int cols = kCols;
    const size_t elems = static_cast<size_t>(rows) * cols;
    const size_t bytes = elems * sizeof(float);
    const size_t traffic_bytes_v12 = elems * sizeof(float) * 4;
    const size_t traffic_bytes_v3 = elems * sizeof(float) * 3;
    const size_t traffic_bytes_v4 = elems * sizeof(float) * 3;
    const size_t flops_v12 = elems * 6;
    const size_t flops_v3 = elems * 7;
    const size_t flops_v4 = elems * 3;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    random_fill(d_x, elems, 2029);

    std::vector<float> h_x(elems), h_ref(elems), h_ref_rmsnorm(elems), h_out(elems);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_x, h_ref, rows, cols);
    cpu_ref_rmsnorm(h_x, h_ref_rmsnorm, rows, cols);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    struct Version {
        const char* name;
        void (*launch)(const float*, float*, int, int);
        size_t traffic_bytes;
        size_t flops;
        const std::vector<float>* ref;
    };

    const std::array<Version, 4> versions{{
        {"naive", launch_naive, traffic_bytes_v12, flops_v12, &h_ref},
        {"v2_block_warp", launch_v2, traffic_bytes_v12, flops_v12, &h_ref},
        {"v3_welford", launch_v3, traffic_bytes_v3, flops_v3, &h_ref},
        {"v4_rmsnorm", launch_v4, traffic_bytes_v4, flops_v4, &h_ref_rmsnorm},
    }};

    for (const auto& version : versions) {
        version.launch(d_x, d_y, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        const float err = max_abs_err(h_out.data(), version.ref->data(), h_out.size());
        if (err > 1e-5f) {
            std::cerr << "correctness failed: max_err=" << err << '\n';
            CUDA_CHECK(cudaFree(d_x));
            CUDA_CHECK(cudaFree(d_y));
            return 1;
        }

        const float ms = timeit([&] { version.launch(d_x, d_y, rows, cols); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row(version.name, ms, version.traffic_bytes, version.flops, err);
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
