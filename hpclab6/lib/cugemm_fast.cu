#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define d_A(i,j) d_A[lda*(i)+(j)]
#define d_B(i,j) d_B[ldb*(i)+(j)]
#define d_C(i,j) d_C[ldc*(i)+(j)]

const int CUGEMM_FAST_KERNEL_SIZE = 8;

__global__ void sgemm_fast_kernel(int k, int m, int n,
    float* d_A, int lda,
    float* d_B, int ldb,
    float* d_C, int ldc)
{
    __shared__ float tiled_A[CUGEMM_FAST_KERNEL_SIZE][CUGEMM_FAST_KERNEL_SIZE],
        tiled_B[CUGEMM_FAST_KERNEL_SIZE][CUGEMM_FAST_KERNEL_SIZE];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int tile_i = threadIdx.x;
    int tile_j = threadIdx.y;
    float d = d_C(i, j);

    for (int po = 0; po < k; po += CUGEMM_FAST_KERNEL_SIZE)
    {
        tiled_A[tile_i][tile_j] = d_A(i, po + tile_j);
        tiled_B[tile_i][tile_j] = d_B(po + tile_i, j);
        __syncthreads();
        for (int pi = 0; pi < CUGEMM_FAST_KERNEL_SIZE; pi++)
        {
            d += tiled_A[tile_i][pi] * tiled_B[pi][tile_j];
        }
        __syncthreads();
    }
    d_C(i, j) += d;
}

void sgemm_fast(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
    float* d_A, * d_B, * d_C;

    cudaMalloc(&d_A, sizeof(float) * m * k);
    cudaMalloc(&d_B, sizeof(float) * k * n);
    cudaMalloc(&d_C, sizeof(float) * m * n);
    cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(float) * m * n);

    dim3 dim_block((m - 1) / CUGEMM_FAST_KERNEL_SIZE + 1, (n - 1) / CUGEMM_FAST_KERNEL_SIZE + 1, 1),
        dim_thread(CUGEMM_FAST_KERNEL_SIZE, CUGEMM_FAST_KERNEL_SIZE, 1);

    sgemm_fast_kernel << <dim_block, dim_thread >> > (k, m, n,
        d_A, lda, d_B, ldb, d_C, ldc);

    cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}