#ifndef _CUGEMM_H_
#define _CUGEMM_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float sgemm_fast(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

void random_initalize_matrix(int M, int N, float* mat);

#define TOLERENCE 1E-1
bool verify_matrix(int M, int N, float* mat1, float* mat2);

void debug_print_matrix(int M, int N, float* mat);
void output_matrix_tofile(const char* filename, int M, int N, float* mat);


// gemm kernel function
__global__ void sgemm_fast_kernel(int k, int lda, int ldb, float* d_C, int ldc, float* d_A, float* d_B);

#endif
