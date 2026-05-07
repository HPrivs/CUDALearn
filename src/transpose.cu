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

constexpr int kTileDim = 32;
constexpr int kRows = 4096;
constexpr int kCols = 4096;

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
// ========= v2: shared memory tiled transpose =========
// shared memory tile 解决的问题：先把输入 tile 暂存下来，
// 再交换 block 坐标写出，让 global load 和 global store 都沿连续地址访问。
__global__ void kernel_v2(const float* a, float* b, int rows, int cols) {    
    __shared__ float smem[kTileDim][kTileDim];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int in_col = blockIdx.x * blockDim.x + tid_x; 
    int in_row = blockIdx.y * blockDim.y + tid_y;

    if (in_col < cols && in_row < rows) {
        smem[tid_y][tid_x] = a[static_cast<size_t>(in_row) * cols + in_col];
    }
    __syncthreads();

    // 块位置调换
    int out_col = blockIdx.y * blockDim.x + tid_x;
    int out_row = blockIdx.x * blockDim.y + tid_y;

    // b的形状是 cols * rows 所以要调换一下
    if (out_col < rows && out_row < cols) {
        b[static_cast<size_t>(out_row) * rows + out_col] = smem[tid_x][tid_y];
    }
}


void launch_v2(const float* a, float* b, int rows, int cols) {
    dim3 block(kTileDim, kTileDim);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));
    kernel_v2<<<grid, block>>>(a, b, rows, cols);

}

// ========= v3: shared memory padding =========
// shared memory padding 解决的问题：给每行多留 1 个 float，
// 让按列读取 tile 时相邻线程更少落到同一个 shared memory bank。
__global__ void kernel_v3(const float* a, float* b, int rows, int cols) {
    __shared__ float smem[kTileDim][kTileDim + 1];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int in_col = blockIdx.x * blockDim.x + tid_x;
    int in_row = blockIdx.y * blockDim.y + tid_y;

    if (in_col < cols && in_row < rows) {
        smem[tid_y][tid_x] = a[static_cast<size_t>(in_row) * cols + in_col];
    }
    __syncthreads();

    int out_col = blockIdx.y * blockDim.x + tid_x;
    int out_row = blockIdx.x * blockDim.y + tid_y;

    if (out_col < rows && out_row < cols) {
        b[static_cast<size_t>(out_row) * rows + out_col] = smem[tid_x][tid_y];
    }
}

void launch_v3(const float* a, float* b, int rows, int cols) {
    dim3 block(kTileDim, kTileDim);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));
    kernel_v3<<<grid, block>>>(a, b, rows, cols);
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

    print_header();

    struct Version {
        const char* name;
        void (*launch)(const float*, float*, int, int);
    };

    const std::array<Version, 3> versions{{
        {"naive", launch_naive},
        {"v2", launch_v2},
        {"v3", launch_v3}
    }};

    for (const auto& version : versions) {
        version.launch(d_a, d_b, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_b, bytes, cudaMemcpyDeviceToHost));
        float err = 0.0f;
        if (!check_close(h_out.data(), h_ref.data(), h_out.size(), 0.0f, 0.0f, &err)) {
            std::cerr << "correctness failed: max_err=" << err << '\n';
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_b));
            return 1;
        }

        float ms = timeit([&] { version.launch(d_a, d_b, rows, cols); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row(version.name, ms, traffic_bytes, flops, err);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
