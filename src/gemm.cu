// GEMM: C[M, N] = A[M, K] * B[K, N]
// C[row, col] = sum_k A[row, k] * B[k, col]
// I/O shape: A is M x K, B is K x N, C is M x N, all row-major
// dtype: float32
// default problem size: M = 512, N = 512, K = 512
// theoretical traffic per output element:
//   naive: read K floats from A + K floats from B + write one C = (2K + 1) * 4B
//   v2_smem with full 16x16 tiles: read about K/16 A + K/16 B + write one C
//     = (2K/16 + 1) * 4B
//   v3_reg_tile with 32x16 output tiles: read about K/16 A + K/32 B + write one C
//     = (K/16 + K/32 + 1) * 4B
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
constexpr int kTileK = 16;
constexpr int kV3RowsPerThread = 2;
constexpr int kV3TileM = kBlockY * kV3RowsPerThread;

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

// ========= v2: shared memory tile =========
// shared memory tiling 解决的问题：把一个输出 tile 内会重复读取的 A/B 片段缓存起来，减少 global memory load。
__global__ void kernel_v2(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[kBlockY][kTileK];
    __shared__ float tile_b[kTileK][kBlockX];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    float sum = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k += kTileK) {
        const int a_col = tile_k + tx;
        const int b_row = tile_k + ty;
        tile_a[ty][tx] = (row < m && a_col < k) ? 
                            a[static_cast<size_t>(row) * k + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < k && col < n) ? 
                            b[static_cast<size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < kTileK; inner++) {
            sum += tile_a[ty][inner] * tile_b[inner][tx];
        }
        __syncthreads();
    }
    
    if (row < m && col < n) {
        c[static_cast<size_t>(row) * n + col] = sum;
    }

} 

void launch_v2(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(div_up(n, block.x), div_up(m, block.y));
    kernel_v2<<<grid, block>>>(a, b, c, m, n, k);
}

// ========= v3: 2x1 register tile =========
// register tile 解决的问题：让一个线程维护多个寄存器累加器，复用一次 shared memory 读取服务多个输出。
__global__ void kernel_v3(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[kV3TileM][kTileK];
    __shared__ float tile_b[kTileK][kBlockX];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.y * kV3TileM + ty;
    int row1 = row0 + kBlockY;
    int col = blockIdx.x * kBlockX + tx;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k += kTileK) {
        const int a_col = tile_k + tx;
        const int b_row = tile_k + ty;

        tile_a[ty][tx] = (row0 < m && a_col < k) ?
                            a[static_cast<size_t>(row0) * k + a_col] : 0.0f;
        tile_a[ty + kBlockY][tx] = (row1 < m && a_col < k) ?
                            a[static_cast<size_t>(row1) * k + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < k && col < n) ?
                            b[static_cast<size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < kTileK; inner++) {
            const float b_val = tile_b[inner][tx];
            sum0 += tile_a[ty][inner] * b_val;
            sum1 += tile_a[ty + kBlockY][inner] * b_val;
        }
        __syncthreads();
    }

    if (row0 < m && col < n) {
        c[static_cast<size_t>(row0) * n + col] = sum0;
    }
    if (row1 < m && col < n) {
        c[static_cast<size_t>(row1) * n + col] = sum1;
    }
}

void launch_v3(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(div_up(n, block.x), div_up(m, kV3TileM));
    kernel_v3<<<grid, block>>>(a, b, c, m, n, k);
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
    const size_t traffic_bytes_v2 =
        (elems_a * static_cast<size_t>(div_up(n, kBlockX)) +
         elems_b * static_cast<size_t>(div_up(m, kBlockY)) +
         elems_c) * sizeof(float);
    const size_t traffic_bytes_v3 =
        (elems_a * static_cast<size_t>(div_up(n, kBlockX)) +
         elems_b * static_cast<size_t>(div_up(m, kV3TileM)) +
         elems_c) * sizeof(float);
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

    const std::array<Version, 3> versions{{
        {"naive", launch_naive, traffic_bytes_naive, flops_naive},
        {"v2_smem", launch_v2, traffic_bytes_v2, flops_naive},
        {"v3_reg_tile", launch_v3, traffic_bytes_v3, flops_naive},
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
