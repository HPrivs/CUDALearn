#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(_err));                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <typename Launcher>
float timeit(Launcher&& launcher, int warmup = 10, int iters = 100) {
    for (int i = 0; i < warmup; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

template <typename T>
void random_fill(T* d_ptr, size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<T> host(n);
    for (size_t i = 0; i < n; ++i) {
        host[i] = static_cast<T>(dist(gen));
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
float max_abs_err(const T* a, const T* b, size_t n) {
    float max_err = 0.0f;
    size_t bad_idx = n;
    for (size_t i = 0; i < n; ++i) {
        float err = std::fabs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (err > max_err) {
            max_err = err;
            bad_idx = i;
        }
    }
    if (bad_idx != n && max_err != 0.0f) {
        std::cout << "first mismatch at idx=" << bad_idx << ", ref=" << b[bad_idx]
                  << ", got=" << a[bad_idx] << ", abs_err=" << max_err << '\n';
    }
    return max_err;
}

inline void print_row(const char* version, float ms, size_t bytes, size_t flops, float err) {
    const double seconds = static_cast<double>(ms) * 1e-3;
    const double gbps = seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1e9 : 0.0;
    const double tflops = seconds > 0.0 ? static_cast<double>(flops) / seconds / 1e12 : 0.0;
    std::cout << std::left << std::setw(12) << version << std::right << std::setw(10)
              << std::fixed << std::setprecision(4) << ms << std::setw(12)
              << std::setprecision(2) << gbps << std::setw(12) << std::setprecision(4)
              << tflops << std::setw(12) << std::setprecision(6) << err << '\n';
}
