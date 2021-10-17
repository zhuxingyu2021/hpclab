#include "cugemm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>
#include <iostream>
#include <omp.h>
#include "mytime.h"
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

float sgemm_fast_multithread(int k, int m, int n,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc,
    int n_thread, int device_cnt)
{
    float timestart, timeend;
    int common_blocksz_x = m / n_thread;
#ifdef WIN64
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * n_thread);
#else
    cudaStream_t streams[n_thread];
#endif

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaDeviceSynchronize());
    timestart = get_wall_time();

#pragma omp parallel for num_threads(n_thread) shared(k, m, n, A, lda, B, ldb, C, ldc, common_blocksz_x, streams) 
        for (int idx_x = 0; idx_x < m; idx_x += common_blocksz_x) {
            int my_rank = omp_get_thread_num();
            int blocksz_x = MIN(m - idx_x, common_blocksz_x);

            float* d_A, * d_B, * d_C;
            size_t pitch_a, pitch_b, pitch_c;

            int d_m = ((blocksz_x - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
            int d_n = ((n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1) * KERNEL_SIZE * REG_TILE_SIZE;
            int d_k = ((k - 1) / KERNEL_SIZE + 1) * KERNEL_SIZE;

            checkCudaErrors(cudaStreamCreate(&streams[my_rank]));

            checkCudaErrors(cudaMallocPitch(&d_A, &pitch_a, sizeof(float) * d_k, d_m));
            checkCudaErrors(cudaMallocPitch(&d_B, &pitch_b, sizeof(float) * d_n, d_k));
            checkCudaErrors(cudaMallocPitch(&d_C, &pitch_c, sizeof(float) * d_n, d_m));

            checkCudaErrors(cudaMemcpy2DAsync(d_A, pitch_a, &A(idx_x, 0), lda * sizeof(float), k * sizeof(float), blocksz_x, cudaMemcpyHostToDevice, streams[my_rank]));
            checkCudaErrors(cudaMemcpy2DAsync(d_B, pitch_b, B, ldb * sizeof(float), n * sizeof(float), k, cudaMemcpyHostToDevice, streams[my_rank]));
            checkCudaErrors(cudaMemset(d_B + k * pitch_b / sizeof(float), 0, (d_k - k) * pitch_b));
            checkCudaErrors(cudaMemset(d_C, 0, d_m * pitch_c));

            int d_lda = pitch_a / sizeof(float);
            int d_ldb = pitch_b / sizeof(float);
            int d_ldc = pitch_c / sizeof(float);

            dim3 dim_block((blocksz_x - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, (n - 1) / (KERNEL_SIZE * REG_TILE_SIZE) + 1, 1),
                dim_thread(KERNEL_SIZE, KERNEL_SIZE, 1);

            launch_kernel(k, d_lda, d_ldb, d_ldc, d_A, d_B, d_C, &dim_block, &dim_thread, &streams[my_rank]);

            checkCudaErrors(cudaMemcpy2DAsync(&C(idx_x, 0), ldc * sizeof(float), d_C, d_ldc * sizeof(float), n * sizeof(float), blocksz_x, cudaMemcpyDeviceToHost, streams[my_rank]));

            checkCudaErrors(cudaFree(d_A));
            checkCudaErrors(cudaFree(d_B));
            checkCudaErrors(cudaFree(d_C));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    checkCudaErrors(cudaSetDevice(0));
    timeend = get_wall_time();

#ifdef WIN64
    free(streams);
#endif

    return timeend - timestart;
}
