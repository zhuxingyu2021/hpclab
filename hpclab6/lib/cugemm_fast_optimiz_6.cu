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

__global__ void sgemm_fast_kernel_optimiz_6(int k, int m, int n,
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
    int Ci = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.x + threadIdx.x * REG_TILE_SIZE;
    int Cj = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.y + threadIdx.y * REG_TILE_SIZE;

    float4 vec_c0_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c0_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c1_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c1_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c2_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c2_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c3_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c3_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c4_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c4_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c5_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c5_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c6_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c6_47 = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 vec_c7_03 = make_float4(0.0, 0.0, 0.0, 0.0), vec_c7_47 = make_float4(0.0, 0.0, 0.0, 0.0);

    float reg_a0, reg_a1, reg_a2, reg_a3, reg_a4, reg_a5, reg_a6, reg_a7;
    float4 vec_b0_3, vec_b4_7;

    for (int po = 0; po < k; po += KERNEL_SIZE)
    {
        float4 vec_gm_a0 = *reinterpret_cast<float4*>(&d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 0));
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 0][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a0.x;
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 1][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a0.y;
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 2][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a0.z;
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 3][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a0.w;
        float4 vec_gm_a1 = *reinterpret_cast<float4*>(&d_A(Bi + (tx % REG_TILE_SIZE) * KERNEL_SIZE + ty, po + (tx / REG_TILE_SIZE) * REG_TILE_SIZE + 4));
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 4][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a1.x;
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 5][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a1.y;
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 6][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a1.z;
        sm_A[(tx / REG_TILE_SIZE) * REG_TILE_SIZE + 7][(tx % REG_TILE_SIZE) * KERNEL_SIZE + ty] = vec_gm_a1.w;

        *reinterpret_cast<float4*>(&sm_B[tx][ty * REG_TILE_SIZE + 0]) = *reinterpret_cast<float4*>(&d_B(po + tx, Cj + 0));
        *reinterpret_cast<float4*>(&sm_B[tx][ty * REG_TILE_SIZE + 4]) = *reinterpret_cast<float4*>(&d_B(po + tx, Cj + 4));

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

            vec_b0_3 = *reinterpret_cast<float4*>(&sm_B[pi][ty * REG_TILE_SIZE + 0]);
            vec_b4_7 = *reinterpret_cast<float4*>(&sm_B[pi][ty * REG_TILE_SIZE + 4]);

            vec_c0_03.x += reg_a0 * vec_b0_3.x; vec_c0_03.y += reg_a0 * vec_b0_3.y; vec_c0_03.z += reg_a0 * vec_b0_3.z; vec_c0_03.w += reg_a0 * vec_b0_3.w;
            vec_c0_47.x += reg_a0 * vec_b4_7.x; vec_c0_47.y += reg_a0 * vec_b4_7.y; vec_c0_47.z += reg_a0 * vec_b4_7.z; vec_c0_47.w += reg_a0 * vec_b4_7.w;

            vec_c1_03.x += reg_a1 * vec_b0_3.x; vec_c1_03.y += reg_a1 * vec_b0_3.y; vec_c1_03.z += reg_a1 * vec_b0_3.z; vec_c1_03.w += reg_a1 * vec_b0_3.w;
            vec_c1_47.x += reg_a1 * vec_b4_7.x; vec_c1_47.y += reg_a1 * vec_b4_7.y; vec_c1_47.z += reg_a1 * vec_b4_7.z; vec_c1_47.w += reg_a1 * vec_b4_7.w;

            vec_c2_03.x += reg_a2 * vec_b0_3.x; vec_c2_03.y += reg_a2 * vec_b0_3.y; vec_c2_03.z += reg_a2 * vec_b0_3.z; vec_c2_03.w += reg_a2 * vec_b0_3.w;
            vec_c2_47.x += reg_a2 * vec_b4_7.x; vec_c2_47.y += reg_a2 * vec_b4_7.y; vec_c2_47.z += reg_a2 * vec_b4_7.z; vec_c2_47.w += reg_a2 * vec_b4_7.w;

            vec_c3_03.x += reg_a3 * vec_b0_3.x; vec_c3_03.y += reg_a3 * vec_b0_3.y; vec_c3_03.z += reg_a3 * vec_b0_3.z; vec_c3_03.w += reg_a3 * vec_b0_3.w;
            vec_c3_47.x += reg_a3 * vec_b4_7.x; vec_c3_47.y += reg_a3 * vec_b4_7.y; vec_c3_47.z += reg_a3 * vec_b4_7.z; vec_c3_47.w += reg_a3 * vec_b4_7.w;

            vec_c4_03.x += reg_a4 * vec_b0_3.x; vec_c4_03.y += reg_a4 * vec_b0_3.y; vec_c4_03.z += reg_a4 * vec_b0_3.z; vec_c4_03.w += reg_a4 * vec_b0_3.w;
            vec_c4_47.x += reg_a4 * vec_b4_7.x; vec_c4_47.y += reg_a4 * vec_b4_7.y; vec_c4_47.z += reg_a4 * vec_b4_7.z; vec_c4_47.w += reg_a4 * vec_b4_7.w;

            vec_c5_03.x += reg_a5 * vec_b0_3.x; vec_c5_03.y += reg_a5 * vec_b0_3.y; vec_c5_03.z += reg_a5 * vec_b0_3.z; vec_c5_03.w += reg_a5 * vec_b0_3.w;
            vec_c5_47.x += reg_a5 * vec_b4_7.x; vec_c5_47.y += reg_a5 * vec_b4_7.y; vec_c5_47.z += reg_a5 * vec_b4_7.z; vec_c5_47.w += reg_a5 * vec_b4_7.w;

            vec_c6_03.x += reg_a6 * vec_b0_3.x; vec_c6_03.y += reg_a6 * vec_b0_3.y; vec_c6_03.z += reg_a6 * vec_b0_3.z; vec_c6_03.w += reg_a6 * vec_b0_3.w;
            vec_c6_47.x += reg_a6 * vec_b4_7.x; vec_c6_47.y += reg_a6 * vec_b4_7.y; vec_c6_47.z += reg_a6 * vec_b4_7.z; vec_c6_47.w += reg_a6 * vec_b4_7.w;

            vec_c7_03.x += reg_a7 * vec_b0_3.x; vec_c7_03.y += reg_a7 * vec_b0_3.y; vec_c7_03.z += reg_a7 * vec_b0_3.z; vec_c7_03.w += reg_a7 * vec_b0_3.w;
            vec_c7_47.x += reg_a7 * vec_b4_7.x; vec_c7_47.y += reg_a7 * vec_b4_7.y; vec_c7_47.z += reg_a7 * vec_b4_7.z; vec_c7_47.w += reg_a7 * vec_b4_7.w;


        }
        __syncthreads();
    }
    *reinterpret_cast<float4*>(&d_C(Ci + 0, Cj + 0)) = vec_c0_03; *reinterpret_cast<float4*>(&d_C(Ci + 0, Cj + 4)) = vec_c0_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 1, Cj + 0)) = vec_c1_03; *reinterpret_cast<float4*>(&d_C(Ci + 1, Cj + 4)) = vec_c1_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 2, Cj + 0)) = vec_c2_03; *reinterpret_cast<float4*>(&d_C(Ci + 2, Cj + 4)) = vec_c2_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 3, Cj + 0)) = vec_c3_03; *reinterpret_cast<float4*>(&d_C(Ci + 3, Cj + 4)) = vec_c3_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 4, Cj + 0)) = vec_c4_03; *reinterpret_cast<float4*>(&d_C(Ci + 4, Cj + 4)) = vec_c4_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 5, Cj + 0)) = vec_c5_03; *reinterpret_cast<float4*>(&d_C(Ci + 5, Cj + 4)) = vec_c5_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 6, Cj + 0)) = vec_c6_03; *reinterpret_cast<float4*>(&d_C(Ci + 6, Cj + 4)) = vec_c6_47;
    *reinterpret_cast<float4*>(&d_C(Ci + 7, Cj + 0)) = vec_c7_03; *reinterpret_cast<float4*>(&d_C(Ci + 7, Cj + 4)) = vec_c7_47;
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
    sgemm_fast_kernel_optimiz_6 << <dim_block, dim_thread >> > (k, m, n,
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