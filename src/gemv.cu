// GEMV: y[row] = sum_j A[row, j] * x[j]
// I/O shape: A is M x K row-major, x is K, y is M
// dtype: float32
// default problem size: M = 4096, K = 4096
// theoretical traffic per output element: read A row K*4B + read x K*4B + write y 4B
// theoretical FLOPs per output element: K multiplies + (K - 1) adds = 2K - 1

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

int div_up(int a, int b) {
    return (a + b - 1) / b;
}

void cpu_ref(const std::vector<float>& a, const std::vector<float>& x,
             std::vector<float>& y, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += static_cast<double>(a[static_cast<size_t>(row) * cols + col]) *
                   static_cast<double>(x[col]);
        }
        y[row] = static_cast<float>(sum);
    }
}

// ========= v1: naive =========
// 为什么这么做：先让一个线程独自完成一行 dot product，建立 GEMV 的正确性基线。
__global__ void kernel_naive(const float* a, const float* x, float* y, int rows,
                             int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        const size_t row_offset = static_cast<size_t>(row) * cols;
        for (int col = 0; col < cols; ++col) {
            sum += a[row_offset + col] * x[col];
        }
        y[row] = sum;
    }
}

void launch_naive(const float* a, const float* x, float* y, int rows, int cols) {
    int grid = div_up(rows, kBlockSize);
    kernel_naive<<<grid, kBlockSize>>>(a, x, y, rows, cols);
}

// ========= v2: block-per-row shared memory reduce =========
// shared memory block reduce 解决的问题：让一个 block 内的多个线程共同计算同一行，
// 再在片上 shared memory 中合并 partial sum，减少单线程串行累加时间。
__global__ void kernel_v2(const float* a, const float* x, float* y, int rows,
                          int cols) {
    __shared__ float smem[kBlockSize];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0.0f;

    if (row < rows) {
        const size_t row_offset = static_cast<size_t>(row) * cols;
        for (int col = tid; col < cols; col += blockDim.x) {
            sum += a[row_offset + col] * x[col];
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

    if (row < rows && tid == 0) {
        y[row] = smem[0];
    }
}

void launch_v2(const float* a, const float* x, float* y, int rows, int cols) {
    kernel_v2<<<rows, kBlockSize>>>(a, x, y, rows, cols);
}

}  // namespace

int main() {
    const int rows = kRows;
    const int cols = kCols;
    const size_t matrix_elems = static_cast<size_t>(rows) * cols;
    const size_t matrix_bytes = matrix_elems * sizeof(float);
    const size_t x_bytes = static_cast<size_t>(cols) * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t traffic_bytes =
        (matrix_elems + matrix_elems + static_cast<size_t>(rows)) * sizeof(float);
    const size_t flops = static_cast<size_t>(rows) * (2 * static_cast<size_t>(cols) - 1);

    float *d_a = nullptr, *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

    random_fill(d_a, matrix_elems, 2026);
    random_fill(d_x, cols, 2027);

    std::vector<float> h_a(matrix_elems), h_x(cols), h_ref(rows), h_out(rows);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, x_bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_a, h_x, h_ref, rows, cols);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    struct Version {
        const char* name;
        void (*launch)(const float*, const float*, float*, int, int);
    };

    const std::array<Version, 2> versions{{
        {"naive", launch_naive},
        {"v2_block_reduce", launch_v2}
    }};

    for (const auto& version : versions) {
        version.launch(d_a, d_x, d_y, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));
        float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());

        float ms = timeit([&] { version.launch(d_a, d_x, d_y, rows, cols); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row(version.name, ms, traffic_bytes, flops, err);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
