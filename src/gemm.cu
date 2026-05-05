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
//   v6_tc_async uses FP16 A/B + FP32 accumulation/output on Tensor Core:
//     A/B use half bytes and a 32x32 output tile; requires sm_80+ and GEMM_ENABLE_SM80_TC
// reported FLOPs per output element:
//   K multiply + K add = 2K FLOPs

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef GEMM_ENABLE_SM80_TC
#include <mma.h>
#endif

#include <array>
#include <cstdint>
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
constexpr int kV4ColsPerThread = 2;
constexpr int kV4TileN = kBlockX * kV4ColsPerThread;
constexpr int kV5RowsPerThread = 4;
constexpr int kV5ColsPerThread = 4;
constexpr int kV5TileM = kBlockY * kV5RowsPerThread;
constexpr int kV5TileN = kBlockX * kV5ColsPerThread;
#ifdef GEMM_ENABLE_SM80_TC
constexpr int kTcBlockM = 32;
constexpr int kTcBlockN = 32;
constexpr int kTcBlockK = 16;
constexpr int kTcWarpM = 16;
constexpr int kTcWarpN = 16;
constexpr int kTcWarpsPerBlock = 4;
constexpr int kTcThreadsPerBlock = 32 * kTcWarpsPerBlock;
#endif

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

#ifdef GEMM_ENABLE_SM80_TC
void cpu_ref_half(const std::vector<__half>& a,
                  const std::vector<__half>& b,
                  std::vector<float>& c,
                  int m,
                  int n,
                  int k) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            double sum = 0.0;
            for (int kk = 0; kk < k; kk++) {
                const double av = static_cast<double>(__half2float(a[static_cast<size_t>(row) * k + kk]));
                const double bv = static_cast<double>(__half2float(b[static_cast<size_t>(kk) * n + col]));
                sum += av * bv;
            }
            c[static_cast<size_t>(row) * n + col] = static_cast<float>(sum);
        }
    }
}
#endif

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

// ========= v5: final 4x4 register tile for sm_61 =========
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

#ifdef GEMM_ENABLE_SM80_TC
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_ca_shared_global_4(void* smem_ptr, const void* gmem_ptr) {
    const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ bool is_aligned_4(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

__device__ void load_tc_tile_async(const __half* a,
                                   const __half* b,
                                   __half* tile_a,
                                   __half* tile_b,
                                   int m,
                                   int n,
                                   int k,
                                   int block_row,
                                   int block_col,
                                   int tile_k) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr int kAPairs = kTcBlockM * kTcBlockK / 2;
    constexpr int kBPairs = kTcBlockK * kTcBlockN / 2;

    for (int pair = tid; pair < kAPairs; pair += kTcThreadsPerBlock) {
        const int elem = pair * 2;
        const int a_row = elem / kTcBlockK;
        const int a_col = elem % kTcBlockK;
        const int global_row = block_row + a_row;
        const int global_col = tile_k + a_col;
        __half* dst = tile_a + elem;
        const __half* src = a + static_cast<size_t>(global_row) * k + global_col;

        if (global_row < m && global_col + 1 < k && is_aligned_4(src) && is_aligned_4(dst)) {
            cp_async_ca_shared_global_4(dst, src);
        } else {
            dst[0] = (global_row < m && global_col < k) ? src[0] : __float2half(0.0f);
            dst[1] = (global_row < m && global_col + 1 < k) ? src[1] : __float2half(0.0f);
        }
    }

    for (int pair = tid; pair < kBPairs; pair += kTcThreadsPerBlock) {
        const int elem = pair * 2;
        const int b_row = elem / kTcBlockN;
        const int b_col = elem % kTcBlockN;
        const int global_row = tile_k + b_row;
        const int global_col = block_col + b_col;
        __half* dst = tile_b + elem;
        const __half* src = b + static_cast<size_t>(global_row) * n + global_col;

        if (global_row < k && global_col + 1 < n && is_aligned_4(src) && is_aligned_4(dst)) {
            cp_async_ca_shared_global_4(dst, src);
        } else {
            dst[0] = (global_row < k && global_col < n) ? src[0] : __float2half(0.0f);
            dst[1] = (global_row < k && global_col + 1 < n) ? src[1] : __float2half(0.0f);
        }
    }
}
#endif

// ========= v6: Tensor Core WMMA + cp.async pipeline =========
// Tensor Core 路径解决的问题：把标量 FP32 FMA 改成 warp-level MMA，并用 cp.async 预取下一块 K tile。
__global__ void kernel_v6_tc_async(const __half* a, const __half* b, float* c, int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    __shared__ __align__(16) __half tile_a[2][kTcBlockM * kTcBlockK];
    __shared__ __align__(16) __half tile_b[2][kTcBlockK * kTcBlockN];
    __shared__ float tile_c[kTcWarpsPerBlock][kTcWarpM * kTcWarpN];

    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int block_row = blockIdx.y * kTcBlockM;
    const int block_col = blockIdx.x * kTcBlockN;
    const int warp_row = block_row + warp_m * kTcWarpM;
    const int warp_col = block_col + warp_n * kTcWarpN;

    wmma::fragment<wmma::matrix_a, kTcWarpM, kTcWarpN, kTcBlockK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kTcWarpM, kTcWarpN, kTcBlockK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kTcWarpM, kTcWarpN, kTcBlockK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    if (k > 0) {
        load_tc_tile_async(a, b, tile_a[0], tile_b[0], m, n, k, block_row, block_col, 0);
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
    }

    int stage = 0;
    for (int tile_k = 0; tile_k < k; tile_k += kTcBlockK) {
        const int next_tile_k = tile_k + kTcBlockK;
        const int next_stage = stage ^ 1;
        if (next_tile_k < k) {
            load_tc_tile_async(a, b, tile_a[next_stage], tile_b[next_stage],
                               m, n, k, block_row, block_col, next_tile_k);
            cp_async_commit_group();
        }

        const __half* warp_tile_a = tile_a[stage] + warp_m * kTcWarpM * kTcBlockK;
        const __half* warp_tile_b = tile_b[stage] + warp_n * kTcWarpN;
        wmma::load_matrix_sync(a_frag, warp_tile_a, kTcBlockK);
        wmma::load_matrix_sync(b_frag, warp_tile_b, kTcBlockN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        if (next_tile_k < k) {
            cp_async_wait_all();
            __syncthreads();
            stage = next_stage;
        }
    }

    wmma::store_matrix_sync(tile_c[warp_id], c_frag, kTcWarpN, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < kTcWarpM * kTcWarpN; idx += 32) {
        const int local_row = idx / kTcWarpN;
        const int local_col = idx % kTcWarpN;
        const int row = warp_row + local_row;
        const int col = warp_col + local_col;
        if (row < m && col < n) {
            c[static_cast<size_t>(row) * n + col] = tile_c[warp_id][idx];
        }
    }
#endif
}

void launch_v6_tc_async(const __half* a, const __half* b, float* c, int m, int n, int k) {
    const dim3 block(32, kTcWarpsPerBlock);
    const dim3 grid(div_up(n, kTcBlockN), div_up(m, kTcBlockM));
    kernel_v6_tc_async<<<grid, block>>>(a, b, c, m, n, k);
}
#endif

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
#ifdef GEMM_ENABLE_SM80_TC
    const size_t traffic_bytes_v6 =
        (elems_a * static_cast<size_t>(div_up(n, kTcBlockN)) +
         elems_b * static_cast<size_t>(div_up(m, kTcBlockM))) * sizeof(__half) +
        elems_c * sizeof(float);
#endif
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

    const std::array<Version, 5> versions{{
        {"naive", launch_naive, traffic_bytes_naive, flops_naive},
        {"v2_smem", launch_v2, traffic_bytes_v2, flops_naive},
        {"v3_reg_tile", launch_v3, traffic_bytes_v3, flops_naive},
        {"v4_2d_reg_tile", launch_v4, traffic_bytes_v4, flops_naive},
        {"v5_final_4x4", launch_v5, traffic_bytes_v5, flops_naive},
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

#ifdef GEMM_ENABLE_SM80_TC
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major >= 8) {
        std::vector<__half> h_a_half(elems_a), h_b_half(elems_b);
        std::vector<float> h_ref_half(elems_c);
        for (size_t i = 0; i < elems_a; i++) {
            h_a_half[i] = __float2half(h_a[i]);
        }
        for (size_t i = 0; i < elems_b; i++) {
            h_b_half[i] = __float2half(h_b[i]);
        }
        cpu_ref_half(h_a_half, h_b_half, h_ref_half, m, n, k);

        __half *d_a_half = nullptr, *d_b_half = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a_half, elems_a * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_b_half, elems_b * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_a_half, h_a_half.data(), elems_a * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_half, h_b_half.data(), elems_b * sizeof(__half), cudaMemcpyHostToDevice));

        launch_v6_tc_async(d_a_half, d_b_half, d_c, m, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
        const float err = max_abs_err(h_out.data(), h_ref_half.data(), h_out.size());
        if (err > 5e-2f) {
            std::cerr << "tensor core correctness failed: max_err=" << err << '\n';
            CUDA_CHECK(cudaFree(d_a_half));
            CUDA_CHECK(cudaFree(d_b_half));
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_b));
            CUDA_CHECK(cudaFree(d_c));
            return 1;
        }

        const float ms = timeit([&] { launch_v6_tc_async(d_a_half, d_b_half, d_c, m, n, k); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("v6_tc_async", ms, traffic_bytes_v6, flops_naive, err);

        CUDA_CHECK(cudaFree(d_a_half));
        CUDA_CHECK(cudaFree(d_b_half));
    } else {
        std::cout << "v6_tc_async skipped: requires runtime GPU sm_80+; current device is sm_"
                  << prop.major << prop.minor << '\n';
    }
#else
    std::cout << "v6_tc_async skipped: compile with "
                 "nvcc -arch=sm_80 -DGEMM_ENABLE_SM80_TC src/gemm.cu -o debugger/gemm_tc\n";
#endif

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
