// GEMM: C[M, N] = A[M, K] * B[K, N]
// C[row, col] = sum_k A[row, k] * B[k, col]
// I/O shape: A is M x K, B is K x N, C is M x N, all row-major
// dtype: float32
// default problem size: M = 512, N = 512, K = 512
// theoretical traffic per output element for naive:
//   read K floats from A + K floats from B + write one C = (2K + 1) * 4B
// reported FLOPs per output element:
//   K multiply + K add = 2K FLOPs

#include "common.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kN = 512;
constexpr int kK = 512;
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

int div_up(int a, int b) {
    return (a + b - 1) / b;
}

void cpu_ref(const std::vector<float>& a,
             const std::vector<float>& b,
             std::vector<float>& c,
             int m,
             int n,
             int k) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            double sum = 0.0;
            for (int kk = 0; kk < k; kk++) {
                const double av = static_cast<double>(a[static_cast<size_t>(row) * k + kk]);
                const double bv = static_cast<double>(b[static_cast<size_t>(kk) * n + col]);
                sum += av * bv;
            }
            c[static_cast<size_t>(row) * n + col] = static_cast<float>(sum);
        }
    }
}

// ========= v1: naive one thread per C element =========
// naive 解决的问题：先把 GEMM 的二维输出映射和 dot product 公式写正确。
__global__ void kernel_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        sum += a[static_cast<size_t>(row) * k + kk] * b[static_cast<size_t>(kk) * n + col];
    }
    c[static_cast<size_t>(row) * n + col] = sum;
}

void launch_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(div_up(n, block.x), div_up(m, block.y));
    kernel_naive<<<grid, block>>>(a, b, c, m, n, k);
}

}  // namespace

int main() {
    const int m = kM;
    const int n = kN;
    const int k = kK;

    const size_t elems_a = static_cast<size_t>(m) * k;
    const size_t elems_b = static_cast<size_t>(k) * n;
    const size_t elems_c = static_cast<size_t>(m) * n;
    const size_t bytes_a = elems_a * sizeof(float);
    const size_t bytes_b = elems_b * sizeof(float);
    const size_t bytes_c = elems_c * sizeof(float);

    const size_t traffic_bytes_naive = elems_c * static_cast<size_t>(2 * k + 1) * sizeof(float);
    const size_t flops_naive = elems_c * static_cast<size_t>(2 * k);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    random_fill(d_a, elems_a, 2031);
    random_fill(d_b, elems_b, 2032);

    std::vector<float> h_a(elems_a), h_b(elems_b), h_ref(elems_c), h_out(elems_c);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost));
    cpu_ref(h_a, h_b, h_ref, m, n, k);

    std::cout << "version | ms | GB/s | TFLOPS | max_err\n";

    struct Version {
        const char* name;
        void (*launch)(const float*, const float*, float*, int, int, int);
        size_t traffic_bytes;
        size_t flops;
    };

    const std::array<Version, 1> versions{{
        {"naive", launch_naive, traffic_bytes_naive, flops_naive},
    }};

    for (const auto& version : versions) {
        version.launch(d_a, d_b, d_c, m, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
        const float err = max_abs_err(h_out.data(), h_ref.data(), h_out.size());
        if (err > 1e-4f) {
            std::cerr << "correctness failed: max_err=" << err << '\n';
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_b));
            CUDA_CHECK(cudaFree(d_c));
            return 1;
        }

        const float ms = timeit([&] { version.launch(d_a, d_b, d_c, m, n, k); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row(version.name, ms, version.traffic_bytes, version.flops, err);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
