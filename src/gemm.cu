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
//   v4_2d_reg_tile with 32x32 output tiles: read about K/32 A + K/32 B + write one C
//     = (K/32 + K/32 + 1) * 4B
//   v5_final_4x4 with 64x64 output tiles: read about K/64 A + K/64 B + write one C
//     = (K/64 + K/64 + 1) * 4B
//   cublas_sgemm uses cuBLAS FP32 SGEMM as a library comparison baseline; reported traffic
//     uses the minimum effective A + B + C bytes because cuBLAS internal tiling is opaque
// reported FLOPs per output element:
//   K multiply + K add = 2K FLOPs

#include "common.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _status = (expr);                                       \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cublas_status_to_string(_status));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

constexpr int kM = 1024;
constexpr int kN = 1024;
constexpr int kK = 1024;
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
constexpr int kTileK = 16;
constexpr int kV3RowsPerThread = 2;
constexpr int kV3TileM = kBlockY * kV3RowsPerThread;
constexpr int kV4ColsPerThread = 2;
constexpr int kV4TileN = kBlockX * kV4ColsPerThread;
constexpr int kV5RowsPerThread = 4;
constexpr int kV5ColsPerThread = 4;
constexpr int kV5TileM = kBlockY * kV5RowsPerThread;
constexpr int kV5TileN = kBlockX * kV5ColsPerThread;

cublasHandle_t g_cublas_handle = nullptr;

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
    __shared__ float smem_a[kBlockY][kTileK];
    __shared__ float smem_b[kTileK][kBlockX];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float sum = 0.0;
    for (int tile_k = 0; tile_k < k; tile_k += kTileK) {
        size_t a_col = tile_k + tx;
        size_t b_row = tile_k + ty;
        smem_a[ty][tx] = (row < m && a_col < k) ? a[static_cast<size_t>(row) * k + a_col] : 0.0f;
        smem_b[ty][tx] = (b_row < k && col < n) ? b[static_cast<size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < kTileK; inner++) {
            sum += smem_a[ty][inner] * smem_b[inner][tx];
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

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row0 = blockIdx.y * kV3TileM + ty;
    int row1 = row0 + kBlockY;
    int col = blockIdx.x * blockDim.x + tx;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k += kTileK) {
        int a_col = tile_k + tx;
        int b_row = tile_k + ty;

        tile_a[ty][tx] = (row0 < m && a_col < k) ? 
                            a[static_cast<size_t>(row0) * k + a_col] : 0.0f;
        tile_a[ty + kBlockY][tx] = (row1 < m && a_col < k) ?
                            a[static_cast<size_t>(row1) * k + a_col] : 0.0f;
        tile_b[ty][tx] = (col < n && b_row < k) ?
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

// ========= v4: 2x2 register tile =========
// 2D register tile 解决的问题：同时扩大输出 tile 的行和列，让一个线程的多个寄存器累加器复用 A 和 B 两侧数据。
__global__ void kernel_v4(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[kV3TileM][kTileK];
    __shared__ float tile_b[kTileK][kV4TileN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row0 = blockIdx.y * kV3TileM + ty;
    const int row1 = row0 + kBlockY;
    const int col0 = blockIdx.x * kV4TileN + tx;
    const int col1 = col0 + kBlockX;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k += kTileK) {
        const int a_col = tile_k + tx;
        const int b_row = tile_k + ty;

        tile_a[ty][tx] = (row0 < m && a_col < k) ?
                            a[static_cast<size_t>(row0) * k + a_col] : 0.0f;
        tile_a[ty + kBlockY][tx] = (row1 < m && a_col < k) ?
                            a[static_cast<size_t>(row1) * k + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < k && col0 < n) ?
                            b[static_cast<size_t>(b_row) * n + col0] : 0.0f;
        tile_b[ty][tx + kBlockX] = (b_row < k && col1 < n) ?
                            b[static_cast<size_t>(b_row) * n + col1] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < kTileK; inner++) {
            const float a0 = tile_a[ty][inner];
            const float a1 = tile_a[ty + kBlockY][inner];
            const float b0 = tile_b[inner][tx];
            const float b1 = tile_b[inner][tx + kBlockX];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < m && col0 < n) {
        c[static_cast<size_t>(row0) * n + col0] = sum00;
    }
    if (row0 < m && col1 < n) {
        c[static_cast<size_t>(row0) * n + col1] = sum01;
    }
    if (row1 < m && col0 < n) {
        c[static_cast<size_t>(row1) * n + col0] = sum10;
    }
    if (row1 < m && col1 < n) {
        c[static_cast<size_t>(row1) * n + col1] = sum11;
    }
}

void launch_v4(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(div_up(n, kV4TileN), div_up(m, kV3TileM));
    kernel_v4<<<grid, block>>>(a, b, c, m, n, k);
}

// ========= v5: final 4x4 register tile =========
// 收官版复用旧技巧：继续扩大 2D register tile，换取更高 A/B 数据复用；代价是更高 register pressure。
__global__ void kernel_v5(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[kV5TileM][kTileK];
    __shared__ float tile_b[kTileK][kV5TileN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row_base = blockIdx.y * kV5TileM + ty;
    const int col_base = blockIdx.x * kV5TileN + tx;

    float sum[kV5RowsPerThread][kV5ColsPerThread] = {};
    for (int tile_k = 0; tile_k < k; tile_k += kTileK) {
        const int a_col = tile_k + tx;
        const int b_row = tile_k + ty;

        #pragma unroll
        for (int i = 0; i < kV5RowsPerThread; i++) {
            const int row = row_base + i * kBlockY;
            tile_a[ty + i * kBlockY][tx] = (row < m && a_col < k) ?
                a[static_cast<size_t>(row) * k + a_col] : 0.0f;
        }

        #pragma unroll
        for (int j = 0; j < kV5ColsPerThread; j++) {
            const int col = col_base + j * kBlockX;
            tile_b[ty][tx + j * kBlockX] = (b_row < k && col < n) ?
                b[static_cast<size_t>(b_row) * n + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < kTileK; inner++) {
            float a_vals[kV5RowsPerThread];
            float b_vals[kV5ColsPerThread];

            #pragma unroll
            for (int i = 0; i < kV5RowsPerThread; i++) {
                a_vals[i] = tile_a[ty + i * kBlockY][inner];
            }
            #pragma unroll
            for (int j = 0; j < kV5ColsPerThread; j++) {
                b_vals[j] = tile_b[inner][tx + j * kBlockX];
            }
            #pragma unroll
            for (int i = 0; i < kV5RowsPerThread; i++) {
                #pragma unroll
                for (int j = 0; j < kV5ColsPerThread; j++) {
                    sum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kV5RowsPerThread; i++) {
        const int row = row_base + i * kBlockY;
        if (row < m) {
            #pragma unroll
            for (int j = 0; j < kV5ColsPerThread; j++) {
                const int col = col_base + j * kBlockX;
                if (col < n) {
                    c[static_cast<size_t>(row) * n + col] = sum[i][j];
                }
            }
        }
    }
}

void launch_v5(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(div_up(n, kV5TileN), div_up(m, kV5TileM));
    kernel_v5<<<grid, block>>>(a, b, c, m, n, k);
}

// ========= cuBLAS baseline: FP32 SGEMM =========
// cuBLAS 基线解决的问题：给手写 CUDA kernel 提供一个同机同规模的库函数性能上界参考。
void launch_cublas(const float* a, const float* b, float* c, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices. Row-major C = A * B is equivalent to
    // column-major C^T = B^T * A^T using the same memory buffers.
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             b,
                             n,
                             a,
                             k,
                             &beta,
                             c,
                             n));
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
    const size_t traffic_bytes_v4 =
        (elems_a * static_cast<size_t>(div_up(n, kV4TileN)) +
         elems_b * static_cast<size_t>(div_up(m, kV3TileM)) +
         elems_c) * sizeof(float);
    const size_t traffic_bytes_v5 =
        (elems_a * static_cast<size_t>(div_up(n, kV5TileN)) +
         elems_b * static_cast<size_t>(div_up(m, kV5TileM)) +
         elems_c) * sizeof(float);
    const size_t traffic_bytes_cublas = bytes_a + bytes_b + bytes_c;
    const size_t flops_naive = elems_c * static_cast<size_t>(2 * k);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    nvtxRangePushA("random_fill");
    random_fill(d_a, elems_a, 2031);
    random_fill(d_b, elems_b, 2032);
    nvtxRangePop();

    std::vector<float> h_a(elems_a), h_b(elems_b), h_ref(elems_c), h_out(elems_c);
    
    nvtxRangePushA("copy_input_to_host");
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    nvtxRangePushA("cpu_ref");
    cpu_ref(h_a, h_b, h_ref, m, n, k);
    nvtxRangePop();

    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_PEDANTIC_MATH));

    print_header();

    struct Version {
        const char* name;
        void (*launch)(const float*, const float*, float*, int, int, int);
        size_t traffic_bytes;
        size_t flops;
    };

    const std::array<Version, 6> versions{{
        {"naive", launch_naive, traffic_bytes_naive, flops_naive},
        {"v2_smem", launch_v2, traffic_bytes_v2, flops_naive},
        {"v3_reg_tile", launch_v3, traffic_bytes_v3, flops_naive},
        {"v4_2d_reg_tile", launch_v4, traffic_bytes_v4, flops_naive},
        {"v5_final_4x4", launch_v5, traffic_bytes_v5, flops_naive},
        {"cublas_sgemm", launch_cublas, traffic_bytes_cublas, flops_naive},
    }};

    for (const auto& version : versions) {
        version.launch(d_a, d_b, d_c, m, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
        float err = 0.0f;
        if (!check_close(h_out.data(), h_ref.data(), h_out.size(), 1e-4f, 1e-5f, &err)) {
            std::cerr << "correctness failed: max_err=" << err << '\n';
            CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
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

    CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
