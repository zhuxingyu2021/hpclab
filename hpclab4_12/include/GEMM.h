#ifndef _GEMM_H_
#define _GEMM_H_

#include <stdlib.h>

#define TOLERENCE 1E-3

#define A(i,j) A[lda*(i)+(j)]
#define B(i,j) B[ldb*(i)+(j)]
#define C(i,j) C[ldc*(i)+(j)]

#define GEMM_CACHELINE_SIZE 64

/******************************************
 * 矩阵乘法中static调度和dynamic调度的实验结论：
 * 1. 对于可以被均匀分块的矩阵，使用static调度
 * 2. 对于不可以被均匀分块的矩阵，使用dynamic调度
 */
//#define SCHEDULE schedule(dynamic, 1)
//#define N_THREADS 8

void random_initialize_matrix(int M, int N, float* mat);
void output_matrix_tofile(const char* filename, int M, int N, float* mat);
bool verify_matrix(int M, int N, float* mat1, float* mat2);
void debug_print_matrix(int M, int N, float* mat);

void* aligned_malloc(size_t required_bytes, size_t alignment);
void aligned_free(void* p);

void sgemm_naive(int k, int m, int n,
                 float* A, int lda,
                 float* B, int ldb,
                 float* C, int ldc, int n_threads);

void sgemm_fast(int k, int m, int n,
                float* A, int lda,
                float* B, int ldb,
                float* C, int ldc, int n_threads);

#endif
