// GEMM FP32 scalar: C[M, N] = A[M, K] * B[K, N]
// C[row, col] = sum_k A[row, k] * B[k, col]
// I/O shape: A is M x K, B is K x N, C is M x N, all row-major
// dtype: float32 input, float32 accumulation, float32 output
// default problem size: M = 1024, N = 1024, K = 1024
// theoretical traffic per output element:
//   naive: read K floats from A + K floats from B + write one C = (2K + 1) * 4B
//   tiled FP32 scalar versions: each 128x128 C tile reloads A/B K tiles from global memory;
//     for divisible sizes, global traffic is grid_m * grid_n * k_tiles *
//     (BM * BK + BK * BN) * 4B + M * N * 4B
// reported FLOPs per output element:
//   K multiply + K add = 2K FLOPs

#include "common.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

cublasHandle_t g_cublas_handle = nullptr;

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

__device__ __forceinline__ bool can_vectorize_row(int ld, int col, int width) {
	return ((ld & 3) == 0) && ((col & 3) == 0) && (col + 3 < width);
}

__device__ __forceinline__ float4 load_float4_or_zero(const float* ptr,
													   int ld,
													   int row,
													   int col,
													   int rows,
													   int cols) {
	float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (row >= rows || col >= cols) {
		return values;
	}

	const float* base = ptr + static_cast<size_t>(row) * ld + col;
	if (can_vectorize_row(ld, col, cols)) {
		return *reinterpret_cast<const float4*>(base);
	}

	values.x = base[0];
	if (col + 1 < cols) {
		values.y = base[1];
	}
	if (col + 2 < cols) {
		values.z = base[2];
	}
	if (col + 3 < cols) {
		values.w = base[3];
	}
	return values;
}

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
	if (col + 1 < cols) {
		base[1] = values.y;
	}
	if (col + 2 < cols) {
		base[2] = values.z;
	}
	if (col + 3 < cols) {
		base[3] = values.w;
	}
}

template <int BM, int BN, int BK, int PAD>
__device__ __forceinline__ void load_tile_scalar(const float* __restrict__ a,
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
	// 256 个线程合作搬一个 A tile: [BM, BK] = 128x8。
	// 每个线程搬 4 个连续 K 方向元素，所以 a_row 覆盖 0..127，a_col_vec 只会是 0 或 4。
	const int a_row = tid / (BK / 4);
	const int a_col_vec = (tid % (BK / 4)) * 4;

	// 同一批线程也合作搬一个 B tile: [BK, BN] = 8x128。
	// 每个线程搬 4 个连续 N 方向元素，所以 b_row 覆盖 0..7，b_col_vec 覆盖 0,4,...,124。
	const int b_row = tid / (BN / 4);
	const int b_col_vec = (tid % (BN / 4)) * 4;

#pragma unroll
	for (int i = 0; i < 4; ++i) {
		const int global_a_row = block_row + a_row;
		const int global_a_col = tile_k + a_col_vec + i;
		smem_a[a_col_vec + i][a_row] =
			(global_a_row < m && global_a_col < k)
				? a[static_cast<size_t>(global_a_row) * k + global_a_col]
				: 0.0f;

		const int global_b_row = tile_k + b_row;
		const int global_b_col = block_col + b_col_vec + i;
		smem_b[b_row][b_col_vec + i] =
			(global_b_row < k && global_b_col < n)
				? b[static_cast<size_t>(global_b_row) * n + global_b_col]
				: 0.0f;
	}
}

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
	// 映射方式和 load_tile_scalar 相同，只是每个线程用一次 float4 完成 4 个连续元素的 global load。
	const int a_row = tid / (BK / 4);
	const int a_col_vec = (tid % (BK / 4)) * 4;
	const int b_row = tid / (BN / 4);
	const int b_col_vec = (tid % (BN / 4)) * 4;

	const float4 av =
		load_float4_or_zero(a, k, block_row + a_row, tile_k + a_col_vec, m, k);
	smem_a[a_col_vec + 0][a_row] = av.x;
	smem_a[a_col_vec + 1][a_row] = av.y;
	smem_a[a_col_vec + 2][a_row] = av.z;
	smem_a[a_col_vec + 3][a_row] = av.w;

	const float4 bv =
		load_float4_or_zero(b, n, tile_k + b_row, block_col + b_col_vec, k, n);
	smem_b[b_row][b_col_vec + 0] = bv.x;
	smem_b[b_row][b_col_vec + 1] = bv.y;
	smem_b[b_row][b_col_vec + 2] = bv.z;
	smem_b[b_row][b_col_vec + 3] = bv.w;
}

template <int BM, int BN, int BK, int TM, int TN, int PAD>
__device__ __forceinline__ void compute_tile(float (&smem_a)[BK][BM + PAD],
											 float (&smem_b)[BK][BN + PAD],
											 int thread_tile_row,
											 int thread_tile_col,
											 float (&acc)[TM][TN]) {
#pragma unroll
	for (int kk = 0; kk < BK; ++kk) {
		// 当前线程负责 C tile 中一个 8x8 小块。
		// 对固定 kk，先取 8 个 A(row, kk) 和 8 个 B(kk, col)，再做 8x8 外积。
		float a_frag[TM];
		float b_frag[TN];

#pragma unroll
		for (int tm = 0; tm < TM; ++tm) {
			a_frag[tm] = smem_a[kk][thread_tile_row + tm];
		}
#pragma unroll
		for (int tn = 0; tn < TN; ++tn) {
			b_frag[tn] = smem_b[kk][thread_tile_col + tn];
		}

#pragma unroll
		for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
			for (int tn = 0; tn < TN; ++tn) {
				// acc[tm][tn] 对应 C[block_row + thread_tile_row + tm,
				//                     block_col + thread_tile_col + tn] 的部分和。
				acc[tm][tn] = __fmaf_rn(a_frag[tm], b_frag[tn], acc[tm][tn]);
			}
		}
	}
}

template <int TM, int TN>
__device__ __forceinline__ void store_tile_scalar(float* __restrict__ c,
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
#pragma unroll
		for (int tn = 0; tn < TN; ++tn) {
			const int col = block_col + thread_tile_col + tn;
			if (row < m && col < n) {
				c[static_cast<size_t>(row) * n + col] = acc[tm][tn];
			}
		}
	}
}

template <int TM, int TN>
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
		// TN=8 时，一行 8 个结果拆成两个 float4 写回；边界处自动退化为标量写。
		store_float4_or_scalar(
			c, n, row, col, m, n,
			make_float4(acc[tm][0], acc[tm][1], acc[tm][2], acc[tm][3]));
		store_float4_or_scalar(
			c, n, row, col + 4, m, n,
			make_float4(acc[tm][4], acc[tm][5], acc[tm][6], acc[tm][7]));
	}
}

// ========= naive: one thread per C element =========
// naive 解决的问题：先建立正确性和性能下界。
__global__ void kernel_naive(const float* __restrict__ a,
							 const float* __restrict__ b,
							 float* __restrict__ c,
							 int m,
							 int n,
							 int k) {
	// naive 的映射最直接：一个 thread 计算一个 C[row, col]。
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= m || col >= n) {
		return;
	}

	float sum = 0.0f;
	for (int kk = 0; kk < k; ++kk) {
		sum = __fmaf_rn(a[static_cast<size_t>(row) * k + kk],
						b[static_cast<size_t>(kk) * n + col], sum);
	}
	c[static_cast<size_t>(row) * n + col] = sum;
}

void launch_naive(const float* a, const float* b, float* c, int m, int n, int k) {
	const dim3 block(16, 16);
	const dim3 grid(div_up(n, block.x), div_up(m, block.y));
	kernel_naive<<<grid, block>>>(a, b, c, m, n, k);
}

// ========= v1: 8x8 thread tile with scalar global load/store =========
// 8x8 thread tile 解决的问题：让每个线程维护 64 个 FP32 累加器，提高每次 shared memory 读取的计算复用。
__global__ void kernel_v1(const float* __restrict__ a,
						  const float* __restrict__ b,
						  float* __restrict__ c,
						  int m,
						  int n,
						  int k) {
	// smem_a 保存当前 K tile 上的 A 子块，布局为 [BK][BM]，便于 compute 阶段按 kk 取连续 row。
	// smem_b 保存当前 K tile 上的 B 子块，布局为 [BK][BN]，便于 compute 阶段按 kk 取连续 col。
	__shared__ float smem_a[kBlockK][kBlockM];
	__shared__ float smem_b[kBlockK][kBlockN];

	const int tid = threadIdx.y * blockDim.x + threadIdx.x;

	// blockIdx 定位这个 block 负责的 C 大 tile 左上角。
	const int block_row = blockIdx.y * kBlockM;
	const int block_col = blockIdx.x * kBlockN;
	
	// threadIdx 定位当前 thread 在 128x128 C tile 内负责的 8x8 小 tile 左上角。
	const int thread_tile_row = threadIdx.y * kThreadM;
	const int thread_tile_col = threadIdx.x * kThreadN;

	// 一个线程维护 8x8=64 个寄存器累加器，最终写回 64 个 C 元素。
	float acc[kThreadM][kThreadN] = {};

	// 沿 K 维每次处理 BK=8 层：先把 A/B 当前 tile 搬到 shared memory，再累加到 acc。
	for (int tile_k = 0; tile_k < k; tile_k += kBlockK) {
		load_tile_scalar<kBlockM, kBlockN, kBlockK, 0>(
			a, b, smem_a, smem_b, m, n, k, tile_k, block_row, block_col, tid);
		__syncthreads();

		compute_tile<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN, 0>(
			smem_a, smem_b, thread_tile_row, thread_tile_col, acc);
		__syncthreads();
	}


	store_tile_scalar<kThreadM, kThreadN>(
		c, m, n, block_row, block_col, thread_tile_row, thread_tile_col, acc);
}

void launch_v1(const float* a, const float* b, float* c, int m, int n, int k) {
	const dim3 block(kBlockDimX, kBlockDimY);
	const dim3 grid(div_up(n, kBlockN), div_up(m, kBlockM));
	kernel_v1<<<grid, block>>>(a, b, c, m, n, k);
}

// ========= v2: float4 global load/store + padded shared memory layout =========
// float4 和 padding 解决的问题：减少 global load/store 指令，并调整 shared memory stride 以降低 bank conflict 风险。
__global__ void kernel_v2(const float* __restrict__ a,
						  const float* __restrict__ b,
						  float* __restrict__ c,
						  int m,
						  int n,
						  int k) {
	// PAD 改变 shared memory 每行 stride，降低多个线程访问同一 kk 时落到同一 bank 的概率。
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

void launch_v2(const float* a, const float* b, float* c, int m, int n, int k) {
	const dim3 block(kBlockDimX, kBlockDimY);
	const dim3 grid(div_up(n, kBlockN), div_up(m, kBlockM));
	kernel_v2<<<grid, block>>>(a, b, c, m, n, k);
}

// ========= v3: software double buffering =========
// double buffering 解决的问题：用两份 shared memory 交替保存当前计算 tile 和下一轮预取 tile，减少主循环同步并尝试重叠访存与计算。
__global__ void kernel_v3(const float* __restrict__ a,
						  const float* __restrict__ b,
						  float* __restrict__ c,
						  int m,
						  int n,
						  int k) {
	// 两套 shared memory 轮流使用：一套用于 compute，另一套预先装入下一段 K tile。
	__shared__ float smem_a[2][kBlockK][kBlockM + kSmemPad];
	__shared__ float smem_b[2][kBlockK][kBlockN + kSmemPad];

	const int tid = threadIdx.y * blockDim.x + threadIdx.x;
	const int block_row = blockIdx.y * kBlockM;
	const int block_col = blockIdx.x * kBlockN;
	const int thread_tile_row = threadIdx.y * kThreadM;
	const int thread_tile_col = threadIdx.x * kThreadN;
	const int k_tiles = (k + kBlockK - 1) / kBlockK;

	float acc[kThreadM][kThreadN] = {};

	// 先装入第 0 个 K tile；主循环从 tile=1 开始，形成 load next + compute current。
	load_tile_float4<kBlockM, kBlockN, kBlockK, kSmemPad>(
		a, b, smem_a[0], smem_b[0], m, n, k, 0, block_row, block_col, tid);
	__syncthreads();

	for (int tile = 1; tile < k_tiles; ++tile) {
		// compute_stage 是上一轮已装好的 tile；load_stage 是本轮要覆盖写入的另一套 buffer。
		const int compute_stage = (tile - 1) & 1;
		const int load_stage = tile & 1;

		load_tile_float4<kBlockM, kBlockN, kBlockK, kSmemPad>(
			a, b, smem_a[load_stage], smem_b[load_stage], m, n, k,
			tile * kBlockK, block_row, block_col, tid);

		compute_tile<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN, kSmemPad>(
			smem_a[compute_stage], smem_b[compute_stage], thread_tile_row,
			thread_tile_col, acc);
		__syncthreads();
	}

	const int last_stage = (k_tiles - 1) & 1;
	compute_tile<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN, kSmemPad>(
		smem_a[last_stage], smem_b[last_stage], thread_tile_row, thread_tile_col, acc);

	store_tile_float4<kThreadM, kThreadN>(
		c, m, n, block_row, block_col, thread_tile_row, thread_tile_col, acc);
}

void launch_v3(const float* a, const float* b, float* c, int m, int n, int k) {
	const dim3 block(kBlockDimX, kBlockDimY);
	const dim3 grid(div_up(n, kBlockN), div_up(m, kBlockM));
	kernel_v3<<<grid, block>>>(a, b, c, m, n, k);
}

void launch_cublas(const float* a, const float* b, float* c, int m, int n, int k) {
	const float alpha = 1.0f;
	const float beta = 0.0f;

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

size_t bytes_naive(int m, int n, int k) {
	return static_cast<size_t>(m) * n * (2 * static_cast<size_t>(k) + 1) * sizeof(float);
}

size_t bytes_tiled(int m, int n, int k) {
	const size_t grid_m = div_up(m, kBlockM);
	const size_t grid_n = div_up(n, kBlockN);
	const size_t k_tiles = div_up(k, kBlockK);
	const size_t load_bytes =
		grid_m * grid_n * k_tiles * (kBlockM * kBlockK + kBlockK * kBlockN) * sizeof(float);
	const size_t store_bytes = static_cast<size_t>(m) * n * sizeof(float);
	return load_bytes + store_bytes;
}

size_t bytes_cublas_min(int m, int n, int k) {
	return (static_cast<size_t>(m) * k + static_cast<size_t>(k) * n +
			static_cast<size_t>(m) * n) *
		   sizeof(float);
}

size_t flops_gemm(int m, int n, int k) {
	return static_cast<size_t>(m) * n * 2ULL * static_cast<size_t>(k);
}

}  // namespace

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

	CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
	CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_PEDANTIC_MATH));

	struct Version {
		const char* name;
		void (*launch)(const float*, const float*, float*, int, int, int);
		size_t bytes;
	};

	const size_t tiled_bytes = bytes_tiled(m, n, k);
	const size_t flops = flops_gemm(m, n, k);
	const std::array<Version, 5> versions = {{
		{"naive", launch_naive, bytes_naive(m, n, k)},
		{"v1_8x8_tile", launch_v1, tiled_bytes},
		{"v2_float4_pad", launch_v2, tiled_bytes},
		{"v3_double_buf", launch_v3, tiled_bytes},
		{"cublas_sgemm", launch_cublas, bytes_cublas_min(m, n, k)},
	}};

	print_header();
	for (const Version& version : versions) {
		CUDA_CHECK(cudaMemset(d_c, 0, elems_c * sizeof(float)));
		version.launch(d_a, d_b, d_c, m, n, k);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(h_got.data(), d_c, elems_c * sizeof(float), cudaMemcpyDeviceToHost));
		float err = 0.0f;
		const bool ok = check_close(h_got.data(), h_ref.data(), elems_c, 1e-3f, 1e-5f, &err);
		if (!ok) {
			CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
			CUDA_CHECK(cudaFree(d_a));
			CUDA_CHECK(cudaFree(d_b));
			CUDA_CHECK(cudaFree(d_c));
			return EXIT_FAILURE;
		}

		const float ms = timeit([&]() { version.launch(d_a, d_b, d_c, m, n, k); });
		print_row(version.name, ms, version.bytes, flops, err);
	}

	CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	return 0;
}
