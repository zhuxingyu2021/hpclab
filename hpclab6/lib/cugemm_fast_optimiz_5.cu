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

#define REG_TILE_SIZE 8

__global__ void sgemm_fast_kernel_optimiz_5(int k, int m, int n,
    float* d_A, float* d_B, float* d_C)
{
    __shared__ float sm_A[KERNEL_SIZE][KERNEL_SIZE * REG_TILE_SIZE],
        sm_B[KERNEL_SIZE][KERNEL_SIZE * REG_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Block所计算的kernel的左上角第一个元素在矩阵C中的位置为(Bi,Bj)
    int Bi = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.x;
    int Bj = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.y;

    //线程所计算的micro kernel的左上角第一个元素在矩阵C中的位置为(Ci, Cj)
    int Ci = Bi + threadIdx.x * REG_TILE_SIZE;
    int Cj = Bj + threadIdx.y * REG_TILE_SIZE;

    float reg_c0_0 = 0.0, reg_c0_1 = 0.0, reg_c0_2 = 0.0, reg_c0_3 = 0.0, reg_c0_4 = 0.0, reg_c0_5 = 0.0, reg_c0_6 = 0.0, reg_c0_7 = 0.0;
    float reg_c1_0 = 0.0, reg_c1_1 = 0.0, reg_c1_2 = 0.0, reg_c1_3 = 0.0, reg_c1_4 = 0.0, reg_c1_5 = 0.0, reg_c1_6 = 0.0, reg_c1_7 = 0.0;
    float reg_c2_0 = 0.0, reg_c2_1 = 0.0, reg_c2_2 = 0.0, reg_c2_3 = 0.0, reg_c2_4 = 0.0, reg_c2_5 = 0.0, reg_c2_6 = 0.0, reg_c2_7 = 0.0;
    float reg_c3_0 = 0.0, reg_c3_1 = 0.0, reg_c3_2 = 0.0, reg_c3_3 = 0.0, reg_c3_4 = 0.0, reg_c3_5 = 0.0, reg_c3_6 = 0.0, reg_c3_7 = 0.0;
    float reg_c4_0 = 0.0, reg_c4_1 = 0.0, reg_c4_2 = 0.0, reg_c4_3 = 0.0, reg_c4_4 = 0.0, reg_c4_5 = 0.0, reg_c4_6 = 0.0, reg_c4_7 = 0.0;
    float reg_c5_0 = 0.0, reg_c5_1 = 0.0, reg_c5_2 = 0.0, reg_c5_3 = 0.0, reg_c5_4 = 0.0, reg_c5_5 = 0.0, reg_c5_6 = 0.0, reg_c5_7 = 0.0;
    float reg_c6_0 = 0.0, reg_c6_1 = 0.0, reg_c6_2 = 0.0, reg_c6_3 = 0.0, reg_c6_4 = 0.0, reg_c6_5 = 0.0, reg_c6_6 = 0.0, reg_c6_7 = 0.0;
    float reg_c7_0 = 0.0, reg_c7_1 = 0.0, reg_c7_2 = 0.0, reg_c7_3 = 0.0, reg_c7_4 = 0.0, reg_c7_5 = 0.0, reg_c7_6 = 0.0, reg_c7_7 = 0.0;

    float reg_a0, reg_a1, reg_a2, reg_a3, reg_a4, reg_a5, reg_a6, reg_a7;
    float reg_b0, reg_b1, reg_b2, reg_b3, reg_b4, reg_b5, reg_b6, reg_b7;


    for (int po = 0; po < k; po += KERNEL_SIZE)
    {
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 0][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 0);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 1][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 1);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 2][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 2);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 3][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 3);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 4][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 4);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 5][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 5);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 6][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 6);
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 7][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty]
            = d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 7);

        sm_B[tx][ty * REG_TILE_SIZE + 0] = d_B(po + tx, Cj + 0);
        sm_B[tx][ty * REG_TILE_SIZE + 1] = d_B(po + tx, Cj + 1);
        sm_B[tx][ty * REG_TILE_SIZE + 2] = d_B(po + tx, Cj + 2);
        sm_B[tx][ty * REG_TILE_SIZE + 3] = d_B(po + tx, Cj + 3);
        sm_B[tx][ty * REG_TILE_SIZE + 4] = d_B(po + tx, Cj + 4);
        sm_B[tx][ty * REG_TILE_SIZE + 5] = d_B(po + tx, Cj + 5);
        sm_B[tx][ty * REG_TILE_SIZE + 6] = d_B(po + tx, Cj + 6);
        sm_B[tx][ty * REG_TILE_SIZE + 7] = d_B(po + tx, Cj + 7);

        __syncthreads();
        for (int pi = 0; pi < KERNEL_SIZE; pi++)
        {
            reg_a0 = sm_A[pi][tx * REG_TILE_SIZE + 0];
            reg_a1 = sm_A[pi][tx * REG_TILE_SIZE + 1];
            reg_a2 = sm_A[pi][tx * REG_TILE_SIZE + 2];
            reg_a3 = sm_A[pi][tx * REG_TILE_SIZE + 3];
            reg_a4 = sm_A[pi][tx * REG_TILE_SIZE + 4];
            reg_a5 = sm_A[pi][tx * REG_TILE_SIZE + 5];
            reg_a6 = sm_A[pi][tx * REG_TILE_SIZE + 6];
            reg_a7 = sm_A[pi][tx * REG_TILE_SIZE + 7];

            reg_b0 = sm_B[pi][ty * REG_TILE_SIZE + 0];
            reg_b1 = sm_B[pi][ty * REG_TILE_SIZE + 1];
            reg_b2 = sm_B[pi][ty * REG_TILE_SIZE + 2];
            reg_b3 = sm_B[pi][ty * REG_TILE_SIZE + 3];
            reg_b4 = sm_B[pi][ty * REG_TILE_SIZE + 4];
            reg_b5 = sm_B[pi][ty * REG_TILE_SIZE + 5];
            reg_b6 = sm_B[pi][ty * REG_TILE_SIZE + 6];
            reg_b7 = sm_B[pi][ty * REG_TILE_SIZE + 7];

            reg_c0_0 += reg_a0 * reg_b0; reg_c0_1 += reg_a0 * reg_b1; reg_c0_2 += reg_a0 * reg_b2; reg_c0_3 += reg_a0 * reg_b3; reg_c0_4 += reg_a0 * reg_b4; reg_c0_5 += reg_a0 * reg_b5; reg_c0_6 += reg_a0 * reg_b6; reg_c0_7 += reg_a0 * reg_b7;
            reg_c1_0 += reg_a1 * reg_b0; reg_c1_1 += reg_a1 * reg_b1; reg_c1_2 += reg_a1 * reg_b2; reg_c1_3 += reg_a1 * reg_b3; reg_c1_4 += reg_a1 * reg_b4; reg_c1_5 += reg_a1 * reg_b5; reg_c1_6 += reg_a1 * reg_b6; reg_c1_7 += reg_a1 * reg_b7;
            reg_c2_0 += reg_a2 * reg_b0; reg_c2_1 += reg_a2 * reg_b1; reg_c2_2 += reg_a2 * reg_b2; reg_c2_3 += reg_a2 * reg_b3; reg_c2_4 += reg_a2 * reg_b4; reg_c2_5 += reg_a2 * reg_b5; reg_c2_6 += reg_a2 * reg_b6; reg_c2_7 += reg_a2 * reg_b7;
            reg_c3_0 += reg_a3 * reg_b0; reg_c3_1 += reg_a3 * reg_b1; reg_c3_2 += reg_a3 * reg_b2; reg_c3_3 += reg_a3 * reg_b3; reg_c3_4 += reg_a3 * reg_b4; reg_c3_5 += reg_a3 * reg_b5; reg_c3_6 += reg_a3 * reg_b6; reg_c3_7 += reg_a3 * reg_b7;
            reg_c4_0 += reg_a4 * reg_b0; reg_c4_1 += reg_a4 * reg_b1; reg_c4_2 += reg_a4 * reg_b2; reg_c4_3 += reg_a4 * reg_b3; reg_c4_4 += reg_a4 * reg_b4; reg_c4_5 += reg_a4 * reg_b5; reg_c4_6 += reg_a4 * reg_b6; reg_c4_7 += reg_a4 * reg_b7;
            reg_c5_0 += reg_a5 * reg_b0; reg_c5_1 += reg_a5 * reg_b1; reg_c5_2 += reg_a5 * reg_b2; reg_c5_3 += reg_a5 * reg_b3; reg_c5_4 += reg_a5 * reg_b4; reg_c5_5 += reg_a5 * reg_b5; reg_c5_6 += reg_a5 * reg_b6; reg_c5_7 += reg_a5 * reg_b7;
            reg_c6_0 += reg_a6 * reg_b0; reg_c6_1 += reg_a6 * reg_b1; reg_c6_2 += reg_a6 * reg_b2; reg_c6_3 += reg_a6 * reg_b3; reg_c6_4 += reg_a6 * reg_b4; reg_c6_5 += reg_a6 * reg_b5; reg_c6_6 += reg_a6 * reg_b6; reg_c6_7 += reg_a6 * reg_b7;
            reg_c7_0 += reg_a7 * reg_b0; reg_c7_1 += reg_a7 * reg_b1; reg_c7_2 += reg_a7 * reg_b2; reg_c7_3 += reg_a7 * reg_b3; reg_c7_4 += reg_a7 * reg_b4; reg_c7_5 += reg_a7 * reg_b5; reg_c7_6 += reg_a7 * reg_b6; reg_c7_7 += reg_a7 * reg_b7;

        }
        __syncthreads();
    }
    d_C(Ci + 0, Cj + 0) += reg_c0_0; d_C(Ci + 0, Cj + 1) += reg_c0_1; d_C(Ci + 0, Cj + 2) += reg_c0_2; d_C(Ci + 0, Cj + 3) += reg_c0_3; d_C(Ci + 0, Cj + 4) += reg_c0_4; d_C(Ci + 0, Cj + 5) += reg_c0_5; d_C(Ci + 0, Cj + 6) += reg_c0_6; d_C(Ci + 0, Cj + 7) += reg_c0_7;
    d_C(Ci + 1, Cj + 0) += reg_c1_0; d_C(Ci + 1, Cj + 1) += reg_c1_1; d_C(Ci + 1, Cj + 2) += reg_c1_2; d_C(Ci + 1, Cj + 3) += reg_c1_3; d_C(Ci + 1, Cj + 4) += reg_c1_4; d_C(Ci + 1, Cj + 5) += reg_c1_5; d_C(Ci + 1, Cj + 6) += reg_c1_6; d_C(Ci + 1, Cj + 7) += reg_c1_7;
    d_C(Ci + 2, Cj + 0) += reg_c2_0; d_C(Ci + 2, Cj + 1) += reg_c2_1; d_C(Ci + 2, Cj + 2) += reg_c2_2; d_C(Ci + 2, Cj + 3) += reg_c2_3; d_C(Ci + 2, Cj + 4) += reg_c2_4; d_C(Ci + 2, Cj + 5) += reg_c2_5; d_C(Ci + 2, Cj + 6) += reg_c2_6; d_C(Ci + 2, Cj + 7) += reg_c2_7;
    d_C(Ci + 3, Cj + 0) += reg_c3_0; d_C(Ci + 3, Cj + 1) += reg_c3_1; d_C(Ci + 3, Cj + 2) += reg_c3_2; d_C(Ci + 3, Cj + 3) += reg_c3_3; d_C(Ci + 3, Cj + 4) += reg_c3_4; d_C(Ci + 3, Cj + 5) += reg_c3_5; d_C(Ci + 3, Cj + 6) += reg_c3_6; d_C(Ci + 3, Cj + 7) += reg_c3_7;
    d_C(Ci + 4, Cj + 0) += reg_c4_0; d_C(Ci + 4, Cj + 1) += reg_c4_1; d_C(Ci + 4, Cj + 2) += reg_c4_2; d_C(Ci + 4, Cj + 3) += reg_c4_3; d_C(Ci + 4, Cj + 4) += reg_c4_4; d_C(Ci + 4, Cj + 5) += reg_c4_5; d_C(Ci + 4, Cj + 6) += reg_c4_6; d_C(Ci + 4, Cj + 7) += reg_c4_7;
    d_C(Ci + 5, Cj + 0) += reg_c5_0; d_C(Ci + 5, Cj + 1) += reg_c5_1; d_C(Ci + 5, Cj + 2) += reg_c5_2; d_C(Ci + 5, Cj + 3) += reg_c5_3; d_C(Ci + 5, Cj + 4) += reg_c5_4; d_C(Ci + 5, Cj + 5) += reg_c5_5; d_C(Ci + 5, Cj + 6) += reg_c5_6; d_C(Ci + 5, Cj + 7) += reg_c5_7;
    d_C(Ci + 6, Cj + 0) += reg_c6_0; d_C(Ci + 6, Cj + 1) += reg_c6_1; d_C(Ci + 6, Cj + 2) += reg_c6_2; d_C(Ci + 6, Cj + 3) += reg_c6_3; d_C(Ci + 6, Cj + 4) += reg_c6_4; d_C(Ci + 6, Cj + 5) += reg_c6_5; d_C(Ci + 6, Cj + 6) += reg_c6_6; d_C(Ci + 6, Cj + 7) += reg_c6_7;
    d_C(Ci + 7, Cj + 0) += reg_c7_0; d_C(Ci + 7, Cj + 1) += reg_c7_1; d_C(Ci + 7, Cj + 2) += reg_c7_2; d_C(Ci + 7, Cj + 3) += reg_c7_3; d_C(Ci + 7, Cj + 4) += reg_c7_4; d_C(Ci + 7, Cj + 5) += reg_c7_5; d_C(Ci + 7, Cj + 6) += reg_c7_6; d_C(Ci + 7, Cj + 7) += reg_c7_7;
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
    sgemm_fast_kernel_optimiz_5 << <dim_block, dim_thread >> > (k, m, n,
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