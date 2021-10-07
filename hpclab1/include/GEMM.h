#ifndef _GEMM_H_
#define _GEMM_H_

#include <stdlib.h>

#define TOLERENCE 1E-3

#define A(i,j) A[lda*(i)+(j)]
#define B(i,j) B[ldb*(i)+(j)]
#define C(i,j) C[ldc*(i)+(j)]

#define GEMM_CACHELINE_SIZE 64
#define GEMM_AVX2ALIGN_SIZE 64

#define BK 256
#define BN 512

#define BM 1024

void random_initalize_matrix(int M, int N, float* mat);
void output_matrix_tofile(const char* filename, int M, int N, float* mat);
bool verify_matrix(int M, int N, float* mat1, float* mat2);
void debug_print_matrix(int M, int N, float* mat);

void* aligned_malloc(size_t required_bytes, size_t alignment);
void aligned_free(void* p);

void gemm_naive_multiply(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

void gemm_fast_multiply(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

#endif
