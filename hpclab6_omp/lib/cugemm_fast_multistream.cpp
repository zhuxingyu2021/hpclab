#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>
#include <iostream>
#include <omp.h>
#include <stdlib.h>

using namespace std;

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 16
#define REG_TILE_SIZE 8
#endif

#ifndef A
#define A(i,j) A[lda*(i)+(j)]
#define B(i,j) B[ldb*(i)+(j)]
#define C(i,j) C[ldc*(i)+(j)]
#endif

#define MIN(i,j) ((i>j)?(j):(i))

float sgemm_fast_multistream(int k, int m, int n,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc,
    int n_stream)
{
    int common_blocksz_x = m / n_stream;
    int device_cnt;
    checkCudaErrors(cudaGetDeviceCount(&device_cnt));

#ifdef WIN64
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * n_stream);
    float** d_A = (float**)malloc(sizeof(float*) * n_stream);
    float** d_B = (float**)malloc(sizeof(float*) * n_stream);
    float** d_C = (float**)malloc(sizeof(float*) * n_stream);
#else
    cudaStream_t streams[n_stream];
    float* d_A[n_stream], * d_B[n_stream], * d_C[n_stream];
#endif

    int my_rank = 0;

    for (int idx_x = 0; idx_x < m; idx_x += common_blocksz_x, my_rank++) {
        int blocksz_x = MIN(m - idx_x, common_blocksz_x);
        size_t pitch_a, pitch_b, pitch_c;

        int d_m = ((blocksz_x - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
        int d_n = ((n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
        int d_k = ((k - 1) / KERNEL_SIZE + 1) * KERNEL_SIZE;

        checkCudaErrors(cudaSetDevice(my_rank % device_cnt));
        checkCudaErrors(cudaStreamCreate(&streams[my_rank]));

        checkCudaErrors(cudaMallocPitch(&d_A[my_rank], &pitch_a, sizeof(float) * d_k, d_m));
        checkCudaErrors(cudaMallocPitch(&d_B[my_rank], &pitch_b, sizeof(float) * d_n, d_k));
        checkCudaErrors(cudaMallocPitch(&d_C[my_rank], &pitch_c, sizeof(float) * d_n, d_m));

        checkCudaErrors(cudaMemcpy2DAsync(d_A[my_rank], pitch_a, &A(idx_x, 0), lda * sizeof(float), k * sizeof(float), blocksz_x, cudaMemcpyHostToDevice, streams[my_rank]));
        checkCudaErrors(cudaMemcpy2DAsync(d_B[my_rank], pitch_b, B, ldb * sizeof(float), n * sizeof(float), k, cudaMemcpyHostToDevice, streams[my_rank]));
        checkCudaErrors(cudaMemsetAsync(d_B[my_rank] + k * pitch_b / sizeof(float), 0, (d_k - k) * pitch_b, streams[my_rank]));
        checkCudaErrors(cudaMemsetAsync(d_C[my_rank], 0, d_m * pitch_c, streams[my_rank]));

        int d_lda = pitch_a / sizeof(float);
        int d_ldb = pitch_b / sizeof(float);
        int d_ldc = pitch_c / sizeof(float);

        dim3 dim_block((blocksz_x - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, (n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, 1),
            dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

        launch_kernel(k, d_lda, d_ldb, d_ldc, d_A[my_rank], d_B[my_rank], d_C[my_rank], &dim_block, &dim_thread, &streams[my_rank]);

        checkCudaErrors(cudaMemcpy2DAsync(&C(idx_x, 0), ldc * sizeof(float), d_C[my_rank], d_ldc * sizeof(float), n * sizeof(float), blocksz_x, cudaMemcpyDeviceToHost, streams[my_rank]));
    }
    
    for(int i = 0; i < n_stream; i++){
        checkCudaErrors(cudaSetDevice(i % device_cnt));
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaFree(d_A[i]));
        checkCudaErrors(cudaFree(d_B[i]));
        checkCudaErrors(cudaFree(d_C[i]));
    }

#ifdef WIN64
    free(streams);
    free(d_A);
    free(d_B);
    free(d_C);
#endif
    return 0;
}
