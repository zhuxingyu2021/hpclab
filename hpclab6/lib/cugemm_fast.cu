#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define d_A(i,j) d_A[k*(i)+(j)]
#define d_B(i,j) d_B[n*(i)+(j)]
#define d_C(i,j) d_C[n*(i)+(j)]

const int KERNEL_SIZE = 16;

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

__global__ void sgemm_fast_kernel_optimiz_1(int k, int m, int n,
    float* d_A, float* d_B, float* d_C)
{
    __shared__ float tiled_A[KERNEL_SIZE][KERNEL_SIZE],
        tiled_B[KERNEL_SIZE][KERNEL_SIZE];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int tilei = threadIdx.x;
    int tilej = threadIdx.y;
    float c = 0;

    for (int po = 0; po < k; po += KERNEL_SIZE)
    {
        tiled_A[tilei][tilej] = d_A(i, po + tilej);
        tiled_B[tilei][tilej] = d_B(po + tilei, j);
        __syncthreads();
        for (int pi = 0; pi < KERNEL_SIZE; pi++)
        {
            c += tiled_A[tilei][pi] * tiled_B[pi][tilej];
        }
        __syncthreads();
    }
    d_C(i, j) += c;
}

__global__ void sgemm_fast_kernel_optimiz_2(int k, int m, int n,
    float* d_A, float* d_B, float* d_C)
{
    __shared__ float tiled_A[KERNEL_SIZE][KERNEL_SIZE],
        tiled_B[KERNEL_SIZE][KERNEL_SIZE];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int tilei = threadIdx.x;
    int tilej = threadIdx.y;
    float c = 0;

    for (int po = 0; po < k; po += KERNEL_SIZE)
    {
        tiled_A[tilej][tilei] = d_A(i, po + tilej);
        tiled_B[tilei][tilej] = d_B(po + tilei, j);
        __syncthreads();
        for (int pi = 0; pi < KERNEL_SIZE; pi++)
        {
            c += tiled_A[pi][tilei] * tiled_B[pi][tilej];
        }
        __syncthreads();
    }
    d_C(i, j) += c;
}

__global__ void sgemm_fast_kernel(int k, int m, int n,
    float* d_A, float* d_B, float* d_C)
{
    __shared__ float tiled_A[KERNEL_SIZE][KERNEL_SIZE],
        tiled_B[KERNEL_SIZE][KERNEL_SIZE * 2];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int tilei = threadIdx.x;
    int tilej = threadIdx.y;
    float c0 = 0.0, c1 = 0.0;
    float b0 = 0.0, b1 = 0.0;

    for (int po = 0; po < k; po += KERNEL_SIZE)
    {
        tiled_A[tilej][tilei] = d_A(i, po + tilej);
        tiled_B[tilei][tilej] = d_B(po + tilei, j);
        tiled_B[tilei][tilej + KERNEL_SIZE] = d_B(po + tilei, j + KERNEL_SIZE);
        __syncthreads();
        for (int pi = 0; pi < KERNEL_SIZE; pi++)
        {
            b0 = tiled_B[pi][tilei * 2];
            b1 = tiled_B[pi][tilei * 2 + 1];
            d += tiled_A[pi][tilei] * tiled_B[pi][tilej];
        }
        __syncthreads();
    }
    d_C(i, j) += d;
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

    assert(KERNEL_SIZE % 4 == 0);

    dim3 dim_block((m - 1) / KERNEL_SIZE + 1, (n - 1) / (KERNEL_SIZE * 2) + 1, 1),
        dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

    cudaEventRecord(start, 0);
    sgemm_fast_kernel << <dim_block, dim_thread >> > (k, m, n,
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