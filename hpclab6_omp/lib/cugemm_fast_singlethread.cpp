#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"
#include <cassert>
#include <iostream>
using namespace std;

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 16
#define REG_TILE_SIZE 8
#endif

#ifndef d_A
#define d_A(i,j) (lda*(i)+(j))
#define d_B(i,j) (ldb*(i)+(j))
#define d_C(i,j) d_C[ldc*(i)+(j)]
#endif

float sgemm_fast(int k, int m, int n,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float* d_A, * d_B, * d_C;
    size_t pitch_a, pitch_b, pitch_c;

    int d_m = ((m - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
    int d_n = ((n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
    int d_k = ((k - 1) / KERNEL_SIZE + 1) * KERNEL_SIZE;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));

    checkCudaErrors(cudaMallocPitch(&d_A, &pitch_a, sizeof(float) * d_k, d_m));
    checkCudaErrors(cudaMallocPitch(&d_B, &pitch_b, sizeof(float) * d_n, d_k));
    checkCudaErrors(cudaMallocPitch(&d_C, &pitch_c, sizeof(float) * d_n, d_m));

    checkCudaErrors(cudaMemcpy2D(d_A, pitch_a, A, lda * sizeof(float), k * sizeof(float), m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_B, pitch_b, B, ldb * sizeof(float), n * sizeof(float), k, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_B + k * pitch_b / sizeof(float), 0, (d_k - k) * pitch_b));
    checkCudaErrors(cudaMemset(d_C, 0, d_m * pitch_c));

    int d_lda = pitch_a / sizeof(float);
    int d_ldb = pitch_b / sizeof(float);
    int d_ldc = pitch_c / sizeof(float);

    dim3 dim_block((m - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, (n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, 1),
        dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

    launch_kernel(k, d_lda, d_ldb, d_ldc, d_A, d_B, d_C, &dim_block, &dim_thread, NULL);

    checkCudaErrors(cudaMemcpy2D(C, ldc * sizeof(float), d_C, d_ldc * sizeof(float), n * sizeof(float), m, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    checkCudaErrors(cudaEventRecord(stop));
    
    float elapsedTime;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    return elapsedTime;
}
