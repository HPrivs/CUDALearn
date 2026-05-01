// LayerNorm: y[row, col] = (x[row, col] - mean(row)) / sqrt(var(row) + eps)
// mean(row) = sum_j x[row, j] / K
// var(row) = sum_j (x[row, j] - mean(row))^2 / K
// I/O shape: x is M x K row-major, y is M x K row-major
// dtype: float32
// default problem size: M = 4096, K = 4096
// theoretical traffic per output element: read x 3 times + write y once = 16B
// reported FLOPs per output element: 6 FLOPs; sqrt/div and row-level scale ops are not counted.

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
constexpr float kEps = 1e-5f;

int div_up(int a, int b) {
    return (a + b - 1) / b;
}

void cpu_ref(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * cols;

        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += static_cast<double>(x[row_offset + col]);
        }
        const double mean = sum / static_cast<double>(cols);

        double sq_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double diff = static_cast<double>(x[row_offset + col]) - mean;
            sq_sum += diff * diff;
        }
        const double var = sq_sum / static_cast<double>(cols);
        const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(kEps));

        for (int col = 0; col < cols; ++col) {
            y[row_offset + col] =
                static_cast<float>((static_cast<double>(x[row_offset + col]) - mean) * inv_std);
        }
    }
}

// ========= v1: naive multi-pass =========
// naive multi-pass 解决的问题：先用最直接的三次行扫描保证 LayerNorm 公式和边界处理正确。
__global__ void kernel_naive(const float* x, float* y, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const size_t row_offset = static_cast<size_t>(row) * cols;

    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum += x[row_offset + col];
    }
    const float mean = sum / static_cast<float>(cols);

    float sq_sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        const float diff = x[row_offset + col] - mean;
        sq_sum += diff * diff;
    }
    const float var = sq_sum / static_cast<float>(cols);
    const float inv_std = 1.0f / sqrtf(var + kEps);

    for (int col = 0; col < cols; ++col) {
        y[row_offset + col] = (x[row_offset + col] - mean) * inv_std;
    }
}

void launch_naive(const float* x, float* y, int rows, int cols) {
    const int grid = div_up(rows, kBlockSize);
    kernel_naive<<<grid, kBlockSize>>>(x, y, rows, cols);
}

}  // namespace

int main() {
    const int rows = kRows;
    const int cols = kCols;
    const size_t elems = static_cast<size_t>(rows) * cols;
    const size_t bytes = elems * sizeof(float);
    const size_t traffic_bytes = elems * sizeof(float) * 4;
    const size_t flops = elems * 6;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    random_fill(d_x, elems, 2029);

    std::vector<float> h_x(elems), h_ref(elems), h_out(elems);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_x, h_ref, rows, cols);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    struct Version {
        const char* name;
        void (*launch)(const float*, float*, int, int);
        size_t traffic_bytes;
        size_t flops;
    };

    const std::array<Version, 1> versions{{
        {"naive", launch_naive, traffic_bytes, flops},
    }};

    for (const auto& version : versions) {
        version.launch(d_x, d_y, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        const float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());
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
