// Reduce Sum: y = sum_i x[i]
// I/O shape: x is a 1D vector with N elements, y is one scalar
// dtype: float32
// default problem size: N = 1 << 22

#include "common.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumElems = (1 << 22);

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

void launch_naive(const float* x, float* y, int n) {
    CUDA_CHECK(cudaMemset(y, 0, sizeof(float)));
    int grid = (n + kBlockSize - 1) / kBlockSize;
    kernel_naive<<<grid, kBlockSize>>>(x, y, n);
}

}  // namespace

int main() {
    const int n = kNumElems;
    const size_t input_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t output_bytes = sizeof(float);
    const size_t traffic_bytes = input_bytes + output_bytes;
    const size_t flops = static_cast<size_t>(n - 1);


    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, output_bytes));

    random_fill(d_x, n, 2028);

    std::vector<float> h_x(n), h_ref(1), h_out(1);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, input_bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_x, h_ref);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    launch_naive(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, output_bytes, cudaMemcpyDeviceToHost));
    float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());

    float ms = timeit([&] { launch_naive(d_x, d_y, n); }, 3, 20);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_row("naive", ms, traffic_bytes, flops, err);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
