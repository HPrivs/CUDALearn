// ElementWise Add: C[i] = A[i] + B[i]
// I/O shape: A, B, C are 1D vectors with N elements
// dtype: float32
// default problem size: N = 1 << 24

#include "common.cuh"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumElems = (1 << 24);

void cpu_ref(const std::vector<float>& a,
             const std::vector<float>& b,
             std::vector<float>& c) {
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

// ========= v1: naive =========
__global__ void kernel_naive(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }

}

void launch_naive(const float* a, const float* b, float* c, int n) {
    int grid = (n + kBlockSize - 1) / kBlockSize;
    kernel_naive<<<grid, kBlockSize>>>(a, b, c, n);
}

// ========= v2: float4 vectorized load/store =========
// 为什么这么做：一次让一个线程处理 4 个连续 float，可减少访存指令条数。
__global__ void kernel_v2_float4(const float4* a,
                                 const float4* b,
                                 float4* c,
                                 int vec_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_n) {
        float4 av = a[idx];
        float4 bv = b[idx];
        c[idx] = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
    }
}

void launch_v2_float4(const float* a, const float* b, float* c, int n) {
    int vec_n = n / 4;
    if (vec_n > 0) {
        int grid = (vec_n + kBlockSize - 1) / kBlockSize;
        kernel_v2_float4<<<grid, kBlockSize>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            vec_n);
    }

    int tail = n - vec_n * 4;
    if (tail > 0) {
        kernel_naive<<<1, tail>>>(a + vec_n * 4, b + vec_n * 4, c + vec_n * 4, tail);
    }
}


}  // namespace

int main() {
    const int n = kNumElems;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t traffic_bytes = static_cast<size_t>(n) * sizeof(float) * 3;
    const size_t flops = static_cast<size_t>(n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    random_fill(d_a, n, 2026);
    random_fill(d_b, n, 2027);

    std::vector<float> h_a(n), h_b(n), h_ref(n), h_out(n);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, bytes, cudaMemcpyDeviceToHost));
    cpu_ref(h_a, h_b, h_ref);

    std::cout << "version            ms        GB/s     TFLOPS     max_err\n";

    launch_v2_float4(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());

    float ms = timeit([&] {launch_v2_float4(d_a, d_b, d_c, n); });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_row("float4", ms, traffic_bytes, flops, err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
