#ifndef _CUGEMM_H_
#define _CUGEMM_H_
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float sgemm_fast(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

void random_initialize_matrix(int M, int N, float* mat);

#define TOLERENCE 1E-1
bool verify_matrix(int M, int N, float* mat1, float* mat2);

void debug_print_matrix(int M, int N, float* mat);
void output_matrix_tofile(const char* filename, int M, int N, float* mat);


#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);

#endif
