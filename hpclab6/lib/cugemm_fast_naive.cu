#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define d_A(i,j) d_A[k*(i)+(j)]
#define d_B(i,j) d_B[n*(i)+(j)]
#define d_C(i,j) d_C[n*(i)+(j)]

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 8
#endif

__global__ void sgemm_fast_kernel_naive(int k, int m, int n,
    float* d_A, float* d_B, float* d_C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    float c = 0;

    for (int p = 0; p < k; p++)
    {
        c += d_A(i, p) * d_B(p, j);
    }
    d_C(i, j) += c;
}

float sgemm_fast(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
    float* d_A, * d_B, * d_C;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_A, sizeof(float) * m * k);
    cudaMalloc(&d_B, sizeof(float) * k * n);
    cudaMalloc(&d_C, sizeof(float) * m * n);
    cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(float) * m * n);

    dim3 dim_block((m - 1) / (KERNEL_SIZE) + 1, (n - 1) / (KERNEL_SIZE) + 1, 1),
        dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

    cudaEventRecord(start, 0);
    sgemm_fast_kernel_naive << <dim_block, dim_thread >> > (k, m, n,
        d_A, d_B, d_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return elapsedTime;
}
