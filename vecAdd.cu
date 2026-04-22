#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iostream>
#include <stdlib.h>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
 

__global__ void Naive_vecAdd(const float* A, const float* B, float* C, int N)
{
    int workIdx = blockDim.x * blockIdx.x + threadIdx.x;

    // 边界检查，向量长度极少数情况下是blockDim.x的整数倍
    if (workIdx < N)
    {
        C[workIdx] = A[workIdx] + B[workIdx];
    }

}

void initRandomVector(float* vec, int N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < N; i++)
        vec[i] = dis(gen);

}

int main() {
    int N = 10000;
    size_t bytes = N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    initRandomVector(h_A, N);
    initRandomVector(h_B, N);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 若blocksize * gridsize < N，那么超出线程数部分元素不会被计算 (可用grid-stride loop解决)
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    Naive_vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl; 

    // 等待内核执行结束
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i])
            std::cout << "VectorAdd[" << i << "] != " << h_C[i] << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
} 