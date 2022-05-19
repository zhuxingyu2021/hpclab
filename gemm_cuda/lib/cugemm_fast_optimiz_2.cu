#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define d_A(i,j) d_A[k*(i)+(j)]
#define d_B(i,j) d_B[n*(i)+(j)]
#define d_C(i,j) d_C[n*(i)+(j)]

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 16
#endif

#define REG_TILE_SIZE 4

//使用microkernel增加计算访存比
__global__ void sgemm_fast_kernel_optimiz_2(int k, int m, int n,
    float* d_A, float* d_B, float* d_C)
{
    __shared__ float sm_A[KERNEL_SIZE * REG_TILE_SIZE][KERNEL_SIZE],
                     sm_B[KERNEL_SIZE][KERNEL_SIZE * REG_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //线程所计算的micro kernel的左上角第一个元素在矩阵C中的位置为(Ci, Cj)
    int Ci = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.x + threadIdx.x * REG_TILE_SIZE;
    int Cj = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.y + threadIdx.y * REG_TILE_SIZE;

    float reg_c0_0 = 0.0, reg_c0_1 = 0.0, reg_c0_2 = 0.0, reg_c0_3 = 0.0;
    float reg_c1_0 = 0.0, reg_c1_1 = 0.0, reg_c1_2 = 0.0, reg_c1_3 = 0.0;
    float reg_c2_0 = 0.0, reg_c2_1 = 0.0, reg_c2_2 = 0.0, reg_c2_3 = 0.0;
    float reg_c3_0 = 0.0, reg_c3_1 = 0.0, reg_c3_2 = 0.0, reg_c3_3 = 0.0;

    float reg_a0, reg_a1, reg_a2, reg_a3;
    float reg_b0, reg_b1, reg_b2, reg_b3;

    for (int po = 0; po < k; po += KERNEL_SIZE)
    {
        sm_A[tx * REG_TILE_SIZE + 0][ty] = d_A(Ci + 0, po + ty);
        sm_A[tx * REG_TILE_SIZE + 1][ty] = d_A(Ci + 1, po + ty);
        sm_A[tx * REG_TILE_SIZE + 2][ty] = d_A(Ci + 2, po + ty);
        sm_A[tx * REG_TILE_SIZE + 3][ty] = d_A(Ci + 3, po + ty);

        sm_B[tx][ty * REG_TILE_SIZE + 0] = d_B(po + tx, Cj + 0);
        sm_B[tx][ty * REG_TILE_SIZE + 1] = d_B(po + tx, Cj + 1);
        sm_B[tx][ty * REG_TILE_SIZE + 2] = d_B(po + tx, Cj + 2);
        sm_B[tx][ty * REG_TILE_SIZE + 3] = d_B(po + tx, Cj + 3);

        __syncthreads();
        for (int pi = 0; pi < KERNEL_SIZE; pi++)
        {
            reg_a0 = sm_A[tx * REG_TILE_SIZE + 0][pi];
            reg_a1 = sm_A[tx * REG_TILE_SIZE + 1][pi];
            reg_a2 = sm_A[tx * REG_TILE_SIZE + 2][pi];
            reg_a3 = sm_A[tx * REG_TILE_SIZE + 3][pi];

            reg_b0 = sm_B[pi][ty * REG_TILE_SIZE + 0];
            reg_b1 = sm_B[pi][ty * REG_TILE_SIZE + 1];
            reg_b2 = sm_B[pi][ty * REG_TILE_SIZE + 2];
            reg_b3 = sm_B[pi][ty * REG_TILE_SIZE + 3];

            reg_c0_0 += reg_a0 * reg_b0; reg_c0_1 += reg_a0 * reg_b1; reg_c0_2 += reg_a0 * reg_b2; reg_c0_3 += reg_a0 * reg_b3;
            reg_c1_0 += reg_a1 * reg_b0; reg_c1_1 += reg_a1 * reg_b1; reg_c1_2 += reg_a1 * reg_b2; reg_c1_3 += reg_a1 * reg_b3;
            reg_c2_0 += reg_a2 * reg_b0; reg_c2_1 += reg_a2 * reg_b1; reg_c2_2 += reg_a2 * reg_b2; reg_c2_3 += reg_a2 * reg_b3;
            reg_c3_0 += reg_a3 * reg_b0; reg_c3_1 += reg_a3 * reg_b1; reg_c3_2 += reg_a3 * reg_b2; reg_c3_3 += reg_a3 * reg_b3;

        }
        __syncthreads();
    }
    
    d_C(Ci + 0, Cj + 0) += reg_c0_0; d_C(Ci + 0, Cj + 1) += reg_c0_1; d_C(Ci + 0, Cj + 2) += reg_c0_2; d_C(Ci + 0, Cj + 3) += reg_c0_3;
    d_C(Ci + 1, Cj + 0) += reg_c1_0; d_C(Ci + 1, Cj + 1) += reg_c1_1; d_C(Ci + 1, Cj + 2) += reg_c1_2; d_C(Ci + 1, Cj + 3) += reg_c1_3;
    d_C(Ci + 2, Cj + 0) += reg_c2_0; d_C(Ci + 2, Cj + 1) += reg_c2_1; d_C(Ci + 2, Cj + 2) += reg_c2_2; d_C(Ci + 2, Cj + 3) += reg_c2_3;
    d_C(Ci + 3, Cj + 0) += reg_c3_0; d_C(Ci + 3, Cj + 1) += reg_c3_1; d_C(Ci + 3, Cj + 2) += reg_c3_2; d_C(Ci + 3, Cj + 3) += reg_c3_3;
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

    dim3 dim_block((m - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, (n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, 1),
        dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

    cudaEventRecord(start, 0);
    sgemm_fast_kernel_optimiz_2 << <dim_block, dim_thread >> > (k, m, n,
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