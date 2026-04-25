// Matrix Transpose: B[col, row] = A[row, col]
// I/O shape: A is M x N, B is N x M, both row-major
// dtype: float32
// default problem size: M = 4096, N = 4096
// theoretical traffic per element: read A[row, col] 4B + write B[col, row] 4B = 8B
// theoretical FLOPs per element: 0 floating-point operations

#include "common.cuh"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <vector>

namespace {

constexpr int kTileDim = 16;
constexpr int kRows = 4096;
constexpr int kCols = 4096;

int div_up(int a, int b) {
    return (a + b - 1) / b;
}

void cpu_ref(const std::vector<float>& a, std::vector<float>& b, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            b[static_cast<size_t>(col) * rows + row] =
                a[static_cast<size_t>(row) * cols + col];
        }
    }
}

// ========= v1: naive =========
// 为什么这么做：先用一个线程搬运一个元素，建立二维索引和正确性基线。
__global__ void kernel_naive(const float* a, float* b, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        b[static_cast<size_t>(col) * rows + row] =
            a[static_cast<size_t>(row) * cols + col];
    }
}

void launch_naive(const float* a, float* b, int rows, int cols) {
    dim3 block(kTileDim, kTileDim);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));
    kernel_naive<<<grid, block>>>(a, b, rows, cols);
}

}  // namespace

int main() {
    const int rows = kRows;
    const int cols = kCols;
    const size_t num_elems = static_cast<size_t>(rows) * cols;
    const size_t bytes = num_elems * sizeof(float);
    const size_t traffic_bytes = num_elems * sizeof(float) * 2;
    const size_t flops = 0;

    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    random_fill(d_a, num_elems, 2026);

    std::vector<float> h_a(num_elems), h_ref(num_elems), h_out(num_elems);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_a, h_ref, rows, cols);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    struct Version {
        const char* name;
        void (*launch)(const float*, float*, int, int);
    };

    const std::array<Version, 1> versions{{
        {"naive", launch_naive},
    }};

    for (const auto& version : versions) {
        version.launch(d_a, d_b, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_b, bytes, cudaMemcpyDeviceToHost));
        float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());

        float ms = timeit([&] { version.launch(d_a, d_b, rows, cols); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row(version.name, ms, traffic_bytes, flops, err);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
