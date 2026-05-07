#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",       \
                         __FILE__, __LINE__, #expr, cudaGetErrorString(_err)); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct BenchmarkStats {
    float min_ms = 0.0f;
    float mean_ms = 0.0f;
    float max_ms = 0.0f;
};

namespace cudalearn_detail {

template <typename T>
inline double to_double(T value) {
    return static_cast<double>(value);
}

inline double to_double(__half value) {
    return static_cast<double>(__half2float(value));
}

template <typename T>
inline T from_float(float value) {
    return static_cast<T>(value);
}

template <>
inline __half from_float<__half>(float value) {
    return __float2half(value);
}

inline void require_positive(const char* name, int value) {
    if (value <= 0) {
        std::fprintf(stderr, "%s must be positive, got %d\n", name, value);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace cudalearn_detail

inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

template <typename Launcher>
BenchmarkStats timeit_stats(Launcher&& launcher,
                            int warmup = 3,
                            int iters = 20,
                            int repeats = 5) {
    if (warmup < 0) {
        std::fprintf(stderr, "warmup must be non-negative, got %d\n", warmup);
        std::exit(EXIT_FAILURE);
    }
    cudalearn_detail::require_positive("iters", iters);
    cudalearn_detail::require_positive("repeats", repeats);

    for (int i = 0; i < warmup; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    BenchmarkStats stats;
    stats.min_ms = std::numeric_limits<float>::max();

    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            launcher();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        const float sample_ms = elapsed_ms / static_cast<float>(iters);
        stats.min_ms = std::min(stats.min_ms, sample_ms);
        stats.max_ms = std::max(stats.max_ms, sample_ms);
        stats.mean_ms += sample_ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    stats.mean_ms /= static_cast<float>(repeats);
    return stats;
}

template <typename Launcher>
float timeit(Launcher&& launcher, int warmup = 3, int iters = 20) {
    return timeit_stats(launcher, warmup, iters, 1).mean_ms;
}

template <typename T>
void random_fill(T* d_ptr,
                 size_t n,
                 unsigned seed,
                 float low = -1.0f,
                 float high = 1.0f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<T> host(n);
    for (size_t i = 0; i < n; ++i) {
        host[i] = cudalearn_detail::from_float<T>(dist(gen));
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
float max_abs_err(const T* got, const T* ref, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double err =
            std::fabs(cudalearn_detail::to_double(got[i]) - cudalearn_detail::to_double(ref[i]));
        if (err > max_err) {
            max_err = err;
        }
    }
    return static_cast<float>(max_err);
}

template <typename T>
bool check_close(const T* got,
                 const T* ref,
                 size_t n,
                 float atol,
                 float rtol,
                 float* out_max_err = nullptr) {
    double max_err = 0.0;
    size_t first_bad = n;

    for (size_t i = 0; i < n; ++i) {
        const double got_v = cudalearn_detail::to_double(got[i]);
        const double ref_v = cudalearn_detail::to_double(ref[i]);
        const double err = std::fabs(got_v - ref_v);
        const double tol = static_cast<double>(atol) + static_cast<double>(rtol) * std::fabs(ref_v);
        max_err = std::max(max_err, err);
        if (first_bad == n && (std::isnan(err) || err > tol)) {
            first_bad = i;
        }
    }

    if (out_max_err != nullptr) {
        *out_max_err = static_cast<float>(max_err);
    }

    if (first_bad != n) {
        const double got_v = cudalearn_detail::to_double(got[first_bad]);
        const double ref_v = cudalearn_detail::to_double(ref[first_bad]);
        const double err = std::fabs(got_v - ref_v);
        const double tol = static_cast<double>(atol) + static_cast<double>(rtol) * std::fabs(ref_v);
        std::cout << "first mismatch at idx=" << first_bad << ", ref=" << ref_v
                  << ", got=" << got_v << ", abs_err=" << err << ", tol=" << tol
                  << ", max_abs_err=" << max_err << '\n';
        return false;
    }

    return true;
}

inline double effective_gbps(float ms, size_t bytes) {
    const double seconds = static_cast<double>(ms) * 1e-3;
    return seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1e9 : 0.0;
}

inline double effective_tflops(float ms, size_t flops) {
    const double seconds = static_cast<double>(ms) * 1e-3;
    return seconds > 0.0 ? static_cast<double>(flops) / seconds / 1e12 : 0.0;
}

inline void print_header() {
    std::cout << "version | ms | GB/s | TFLOPS | max_err\n";
}

inline void print_row(const char* version, float ms, size_t bytes, size_t flops, float err) {
    std::cout << std::left << std::setw(18) << version << std::right << std::setw(10)
              << std::fixed << std::setprecision(4) << ms << std::setw(12)
              << std::setprecision(2) << effective_gbps(ms, bytes) << std::setw(12)
              << std::setprecision(4) << effective_tflops(ms, flops) << std::setw(12)
              << std::setprecision(6) << err << '\n';
}

inline void print_device_info(int device = -1) {
    if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "device | " << prop.name << " | sm_" << prop.major << prop.minor
              << " | global_mem_GB=" << std::fixed << std::setprecision(2)
              << static_cast<double>(prop.totalGlobalMem) / 1e9
              << " | shared_mem_per_block_KB="
              << static_cast<double>(prop.sharedMemPerBlock) / 1024.0 << '\n';
}
