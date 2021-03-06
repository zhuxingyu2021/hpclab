#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"
#include <cassert>
#include <iostream>
using namespace std;

typedef texture<float4, cudaTextureType1D, cudaReadModeElementType> floatTex;

floatTex texA(0, cudaFilterModePoint, cudaAddressModeBorder);
floatTex texB(0, cudaFilterModePoint, cudaAddressModeBorder);

#define d_A(i,j) (lda*(i)+(j))
#define d_B(i,j) (ldb*(i)+(j))
#define d_C(i,j) d_C[ldc*(i)+(j)]

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 16
#endif

#define REG_TILE_SIZE 8

__global__ void sgemm_fast_kernel(int k, int lda, int ldb, float* d_C, int ldc)
{
    __shared__ float sm_A[KERNEL_SIZE][KERNEL_SIZE * REG_TILE_SIZE],
        sm_B[KERNEL_SIZE][KERNEL_SIZE * REG_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Block所计算的kernel的左上角第一个元素在矩阵C中的位置为(Bi,Bj)
    int Bi = KERNEL_SIZE * REG_TILE_SIZE * blockIdx.x;

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
        float4 vec_gm_a0 = tex1Dfetch(texA, d_A(Bi + (tx % 8) * 16 + ty, po / 4 + (tx / 8) * 2 + 0));
        sm_A[(tx / 8) * 8 + 0][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a0.x;
        sm_A[(tx / 8) * 8 + 1][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a0.y;
        sm_A[(tx / 8) * 8 + 2][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a0.z;
        sm_A[(tx / 8) * 8 + 3][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a0.w;
        float4 vec_gm_a1 = tex1Dfetch(texA, d_A(Bi + (tx % 8) * 16 + ty, po / 4 + (tx / 8) * 2 + 1));
        sm_A[(tx / 8) * 8 + 4][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a1.x;
        sm_A[(tx / 8) * 8 + 5][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a1.y;
        sm_A[(tx / 8) * 8 + 6][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a1.z;
        sm_A[(tx / 8) * 8 + 7][(tx % 8) * 8 + (ty / 8) * 4 + ty % 4 + ((ty / 4) % 2) * 64] = vec_gm_a1.w;

        *reinterpret_cast<float4*>(&sm_B[tx][ty * REG_TILE_SIZE / 2]) = tex1Dfetch(texB, d_B(po + tx, Cj / 4 + 0));
        *reinterpret_cast<float4*>(&sm_B[tx][ty * REG_TILE_SIZE / 2 + KERNEL_SIZE * REG_TILE_SIZE / 2])
            = tex1Dfetch(texB, d_B(po + tx, Cj / 4 + 1));

        __syncthreads();
#pragma unroll
        for (int pi = 0; pi < KERNEL_SIZE; pi++)
        {
            reg_a0 = sm_A[pi][tx * REG_TILE_SIZE / 2 + 0];
            reg_a1 = sm_A[pi][tx * REG_TILE_SIZE / 2 + 1];
            reg_a2 = sm_A[pi][tx * REG_TILE_SIZE / 2 + 2];
            reg_a3 = sm_A[pi][tx * REG_TILE_SIZE / 2 + 3];
            reg_a4 = sm_A[pi][tx * REG_TILE_SIZE / 2 + KERNEL_SIZE * REG_TILE_SIZE / 2 + 0];
            reg_a5 = sm_A[pi][tx * REG_TILE_SIZE / 2 + KERNEL_SIZE * REG_TILE_SIZE / 2 + 1];
            reg_a6 = sm_A[pi][tx * REG_TILE_SIZE / 2 + KERNEL_SIZE * REG_TILE_SIZE / 2 + 2];
            reg_a7 = sm_A[pi][tx * REG_TILE_SIZE / 2 + KERNEL_SIZE * REG_TILE_SIZE / 2 + 3];

            vec_b0_3 = *reinterpret_cast<float4*>(&sm_B[pi][ty * REG_TILE_SIZE / 2]);
            vec_b4_7 = *reinterpret_cast<float4*>(&sm_B[pi][ty * REG_TILE_SIZE / 2 + KERNEL_SIZE * REG_TILE_SIZE / 2]);

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
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    size_t pitch_a, pitch_b, pitch_c;

    int d_m = ((m - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
    int d_n = ((n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
    int d_k = ((k - 1) / KERNEL_SIZE + 1) * KERNEL_SIZE;

    checkCudaErrors(cudaMallocPitch(&d_A, &pitch_a, sizeof(float) * d_k, d_m));
    checkCudaErrors(cudaMallocPitch(&d_B, &pitch_b, sizeof(float) * d_n, d_k));
    checkCudaErrors(cudaMallocPitch(&d_C, &pitch_c, sizeof(float) * d_n, d_m));

    checkCudaErrors(cudaMemcpy2D(d_A, pitch_a, A, lda * sizeof(float), k * sizeof(float), m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_B, pitch_b, B, ldb * sizeof(float), n * sizeof(float), k, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_B + k * pitch_b / sizeof(float), 0, (d_k - k) * pitch_b));
    checkCudaErrors(cudaMemset(d_C, 0, d_m * pitch_c));

    cudaChannelFormatDesc channelDesc_A = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc channelDesc_B = cudaCreateChannelDesc<float4>();
    size_t offset_A, offset_B;
    checkCudaErrors(cudaBindTexture(&offset_A, &texA, d_A, &channelDesc_A, m * pitch_a));
    checkCudaErrors(cudaBindTexture(&offset_B, &texB, d_B, &channelDesc_B, k * pitch_b));

    int d_lda = pitch_a / sizeof(float4);
    int d_ldb = pitch_b / sizeof(float4);
    int d_ldc = pitch_c / sizeof(float);

    dim3 dim_block((m - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, (n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, 1),
        dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    sgemm_fast_kernel << <dim_block, dim_thread >> > (k, d_lda, d_ldb, d_C, d_ldc);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

    checkCudaErrors(cudaMemcpy2D(C, ldc * sizeof(float), d_C, d_ldc * sizeof(float), n * sizeof(float), m, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaUnbindTexture(&texA));
    checkCudaErrors(cudaUnbindTexture(&texB));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return elapsedTime;
}
