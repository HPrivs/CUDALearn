// GEMM FP32 scalar v2: C[M, N] = A[M, K] * B[K, N]
// C[row, col] = sum_k A[row, k] * B[k, col]
// I/O shape: A is M x K, B is K x N, C is M x N, all row-major
// dtype: float32 input, float32 accumulation, float32 output
// default problem size: M = 1024, N = 1024, K = 1024
// theoretical global memory traffic:
//   each 128x128 C tile reloads A/B K tiles from global memory;
//   for divisible sizes, global traffic is grid_m * grid_n * k_tiles *
//   (BM * BK + BK * BN) * 4B + M * N * 4B
// reported FLOPs per output element:
//   K multiply + K add = 2K FLOPs

#include "common.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kM = 1024;
constexpr int kN = 1024;
constexpr int kK = 1024;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;
// 一个 block 负责 128x128 个 C 元素；一个 thread 负责其中 8x8 个 C 元素。
// 因此 block 内线程布局是 16x16，共 256 个线程。
constexpr int kBlockDimX = kBlockN / kThreadN;
constexpr int kBlockDimY = kBlockM / kThreadM;
constexpr int kSmemPad = 4;

// CPU reference，用 double 累加减少参考答案自身的舍入误差。
// a: host 端 A 矩阵，形状 [m, k]，row-major。
// b: host 端 B 矩阵，形状 [k, n]，row-major。
// c: host 端输出 C 矩阵，形状 [m, n]，row-major。
// m/n/k: GEMM 的三个维度，计算 C[m, n] = A[m, k] * B[k, n]。
void cpu_ref(const std::vector<float>& a,
             const std::vector<float>& b,
             std::vector<float>& c,
             int m,
             int n,
             int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                const double av = static_cast<double>(a[static_cast<size_t>(row) * k + kk]);
                const double bv = static_cast<double>(b[static_cast<size_t>(kk) * n + col]);
                sum += av * bv;
            }
            c[static_cast<size_t>(row) * n + col] = static_cast<float>(sum);
        }
    }
}

// 判断从一行矩阵里以 col 为起点读取 4 个 float 是否安全。
// ld: 当前矩阵的 leading dimension，也就是 row-major 下每行元素数。
// col: 本次访问的起始列。
// width: 当前矩阵实际列数，用来判断 col..col+3 是否越界。
__device__ __forceinline__ bool can_vectorize_row(int ld, int col, int width) {
    return ((ld & 3) == 0) && ((col & 3) == 0) && (col + 3 < width);
}

// 从 row-major 矩阵中读取连续 4 个 float；边界处自动补 0。
// ptr: 矩阵起始地址。
// ld: leading dimension，row-major 下等于矩阵每行元素数。
// row/col: 读取起点，语义是 ptr[row, col..col+3]。
// rows/cols: 矩阵实际形状，用于处理非整除边界。
__device__ __forceinline__ float4 load_float4_or_zero(const float* ptr,
                                                        int ld,
                                                        int row,
                                                        int col,
                                                        int rows,
                                                        int cols) {

    float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (row >= rows || col >= cols)
        return values;

    const float* base = ptr + static_cast<size_t>(row) * ld + col;
    if (can_vectorize_row(ld, col, cols)) {
        return  *reinterpret_cast<const float4*>(base);
    }

    values.x = base[0];
    if (col + 1 < cols)
        values.y = base[1];
    if (col + 2 < cols)
        values.z = base[2];
    if (col + 3 < cols) {
        values.w = base[3];
    }

    return values;

}

// 向 row-major 矩阵写入连续 4 个 float；边界或未对齐时退化成标量写。
// ptr: 矩阵起始地址。
// ld: leading dimension，row-major 下等于矩阵每行元素数。
// row/col: 写入起点，语义是 ptr[row, col..col+3]。
// rows/cols: 矩阵实际形状，用于避免越界写。
// values: 要写入的 4 个结果。
__device__ __forceinline__ void store_float4_or_scalar(float* ptr,
                                                        int ld,
                                                        int row,
                                                        int col,
                                                        int rows,
                                                        int cols,
                                                        float4 values) {
    if (row >= rows || col >= cols) {
        return;
    }

    float* base = ptr + static_cast<size_t>(row) * ld + col;
    if (can_vectorize_row(ld, col, cols)) {
        *reinterpret_cast<float4*>(base) = values;
        return;
    }

    base[0] = values.x;
    if (col + 1 < cols)
        base[1] = values.y;
    if (col + 2 < cols)
        base[2] = values.z;
    if (col + 3 < cols)
        base[3] = values.w;

}
// 把当前 K tile 的 A/B 子块从 global memory 搬到 shared memory。
// BM/BN/BK: 一个 CTA 负责的 C tile 形状 [BM, BN]，每次沿 K 维处理 BK 层。
// PAD: shared memory 每行额外 padding，降低 bank conflict 风险。
// a/b: device 端输入矩阵 A[m, k] 和 B[k, n]。
// smem_a/smem_b: shared memory tile，布局分别是 [BK][BM + PAD] 和 [BK][BN + PAD]。
// m/n/k: GEMM 的三个维度。
// tile_k: 当前正在处理的 K 维 tile 起点。
// block_row/block_col: 当前 CTA 负责的 C tile 左上角坐标。
// tid: CTA 内线性线程号，用来分配每个线程搬哪一段 float4。
template <int BM, int BN, int BK, int PAD>
__device__ __forceinline__ void load_tile_float4(const float* __restrict__ a,
                                                 const float* __restrict__ b,
                                                 float (&smem_a)[BK][BM + PAD],
                                                 float (&smem_b)[BK][BN + PAD],
                                                 int m,
                                                 int n,
                                                 int k,
                                                 int tile_k,
                                                 int block_row,
                                                 int block_col,
                                                 int tid) {
    int a_row = tid / (BK / 4);
    int a_col_vec = (tid % (BK / 4)) * 4;
    int b_row = tid / (BN / 4);
    int b_col_vec = (tid % (BN / 4)) * 4;

    const float4 av = load_float4_or_zero(a, k, block_row + a_row, tile_k + a_col_vec, m, k);
    smem_a[a_col_vec + 0][a_row] = av.x;
    smem_a[a_col_vec + 1][a_row] = av.y;
    smem_a[a_col_vec + 2][a_row] = av.z;
    smem_a[a_col_vec + 3][a_row] = av.w;

    const float4 bv = load_float4_or_zero(b, n, tile_k + b_row, block_col + b_col_vec, k, n);
    smem_b[b_row][b_col_vec + 0] = bv.x;
    smem_b[b_row][b_col_vec + 1] = bv.y;
    smem_b[b_row][b_col_vec + 2] = bv.z;
    smem_b[b_row][b_col_vec + 3] = bv.w;
}

// 使用 shared memory 中的一个 K tile，累加当前线程负责的 TM x TN 输出小块。
// BM/BN/BK/PAD: shared memory tile 的形状参数，需和 load_tile_float4 保持一致。
// TM/TN: 每个线程负责的 C 子块大小；本文件中为 8x8。
// smem_a/smem_b: 已经装入 shared memory 的 A/B tile。
// thread_tile_row/thread_tile_col: 当前线程负责的小块在 CTA 的 C tile 内的左上角。
// acc: 当前线程私有的寄存器累加器，形状 [TM][TN]。
template<int BM, int BN, int BK, int TM, int TN, int PAD>
__device__ __forceinline__ void compute_tile(float (&smem_a)[BK][BM + PAD],
                                             float (&smem_b)[BK][BN + PAD],
                                             int thread_tile_row,
                                             int thread_tile_col,
                                             float (&acc)[TM][TN]) {
    
    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
        float a_frag[TM];
        float b_frag[TN];

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            a_frag[tm] = smem_a[kk][thread_tile_row + tm];
        }
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            b_frag[tn] = smem_b[kk][thread_tile_col + tn];
        }

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                acc[tm][tn] = __fmaf_rn(a_frag[tm], b_frag[tn], acc[tm][tn]);
            }
        }
    }
}

template <int TM, int TN>
// 把当前线程的 TM x TN 寄存器结果写回 global memory。
// TM/TN: 每个线程负责的输出子块大小；本文件中 TN=8，所以每行拆成两个 float4。
// c: device 端输出矩阵 C[m, n]。
// m/n: C 的实际形状，用于处理边界。
// block_row/block_col: 当前 CTA 负责的 C tile 左上角坐标。
// thread_tile_row/thread_tile_col: 当前线程负责的小块在 CTA 的 C tile 内的左上角。
// acc: 当前线程私有的寄存器累加器。
__device__ __forceinline__ void store_tile_float4(float* __restrict__ c,
                                                  int m,
                                                  int n,
                                                  int block_row,
                                                  int block_col,
                                                  int thread_tile_row,
                                                  int thread_tile_col,
                                                  float (&acc)[TM][TN]) {
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int row = block_row + thread_tile_row + tm;
        const int col = block_col + thread_tile_col;
        store_float4_or_scalar(
            c, n, row, col, m, n,
            make_float4(acc[tm][0], acc[tm][1], acc[tm][2], acc[tm][3]));
        store_float4_or_scalar(
            c, n, row, col + 4, m, n,
            make_float4(acc[tm][4], acc[tm][5], acc[tm][6], acc[tm][7]));
    }
}

// ========= v2: float4 global load/store + padded shared memory layout =========
// float4 和 padding 解决的问题：减少 global load/store 指令，并调整 shared memory stride 以降低 bank conflict 风险。
// a/b/c: device 端矩阵 A[m, k]、B[k, n]、C[m, n]，全部 row-major。
// m/n/k: GEMM 的三个维度，支持非 128 或 8 整除的边界规模。
__global__ void kernel_v2(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c,
                          int m,
                          int n,
                          int k) {
    __shared__ float smem_a[kBlockK][kBlockM + kSmemPad];
    __shared__ float smem_b[kBlockK][kBlockN + kSmemPad];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;
    const int thread_tile_row = threadIdx.y * kThreadM;
    const int thread_tile_col = threadIdx.x * kThreadN;

    float acc[kThreadM][kThreadN] = {};
    for (int tile_k = 0; tile_k < k; tile_k += kBlockK) {
        load_tile_float4<kBlockM, kBlockN, kBlockK, kSmemPad>(
            a, b, smem_a, smem_b, m, n, k, tile_k, block_row, block_col, tid);
        __syncthreads();

        compute_tile<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN, kSmemPad>(
            smem_a, smem_b, thread_tile_row, thread_tile_col, acc);
        __syncthreads();
    }

    store_tile_float4<kThreadM, kThreadN>(
        c, m, n, block_row, block_col, thread_tile_row, thread_tile_col, acc);
}

// host 端 launch wrapper，统一 kernel 调用签名，方便后续放进 benchmark 表。
// a/b/c: device 端矩阵 A[m, k]、B[k, n]、C[m, n]。
// m/n/k: GEMM 的三个维度。
void launch_v2(const float* a, const float* b, float* c, int m, int n, int k) {
    const dim3 block(kBlockDimX, kBlockDimY);
    const dim3 grid(div_up(n, kBlockN), div_up(m, kBlockM));
    kernel_v2<<<grid, block>>>(a, b, c, m, n, k);
}

// 估算 v2 的有效 global memory 访存字节数，用于 print_row 的 GB/s。
// m/n/k: GEMM 的三个维度；非整除时按实际 grid tile 数估算。
size_t bytes_tiled(int m, int n, int k) {
    const size_t grid_m = div_up(m, kBlockM);
    const size_t grid_n = div_up(n, kBlockN);
    const size_t k_tiles = div_up(k, kBlockK);
    const size_t load_bytes =
        grid_m * grid_n * k_tiles * (kBlockM * kBlockK + kBlockK * kBlockN) * sizeof(float);
    const size_t store_bytes = static_cast<size_t>(m) * n * sizeof(float);
    return load_bytes + store_bytes;
}

// GEMM 有效 FLOPs：每个 C 元素做 k 次乘法和 k 次加法，计 2*k FLOPs。
// m/n/k: GEMM 的三个维度。
size_t flops_gemm(int m, int n, int k) {
    return static_cast<size_t>(m) * n * 2ULL * static_cast<size_t>(k);
}

}  // namespace

// 命令行入口。
// argc/argv: 不传参数时使用默认 1024x1024x1024；传 3 个参数时按 [M N K] 运行。
int main(int argc, char** argv) {
    print_device_info();

    int m = kM;
    int n = kN;
    int k = kK;
    if (argc == 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [M N K]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        std::fprintf(stderr, "M, N and K must be positive\n");
        return EXIT_FAILURE;
    }

    const size_t elems_a = static_cast<size_t>(m) * k;
    const size_t elems_b = static_cast<size_t>(k) * n;
    const size_t elems_c = static_cast<size_t>(m) * n;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, elems_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, elems_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, elems_c * sizeof(float)));

    random_fill(d_a, elems_a, 2026, -0.5f, 0.5f);
    random_fill(d_b, elems_b, 2027, -0.5f, 0.5f);

    std::vector<float> h_a(elems_a);
    std::vector<float> h_b(elems_b);
    std::vector<float> h_ref(elems_c);
    std::vector<float> h_got(elems_c);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, elems_a * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, elems_b * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_ref(h_a, h_b, h_ref, m, n, k);

    CUDA_CHECK(cudaMemset(d_c, 0, elems_c * sizeof(float)));
    launch_v2(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_got.data(), d_c, elems_c * sizeof(float), cudaMemcpyDeviceToHost));
    float err = 0.0f;
    const bool ok = check_close(h_got.data(), h_ref.data(), elems_c, 1e-3f, 1e-5f, &err);

    print_header();
    if (ok) {
        const float ms = timeit([&]() { launch_v2(d_a, d_b, d_c, m, n, k); });
        print_row("v2_float4_pad", ms, bytes_tiled(m, n, k), flops_gemm(m, n, k), err);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
