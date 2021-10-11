#ifndef _CUGEMM_H_
#define _CUGEMM_H_

void sgemm_fast(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

void random_initalize_matrix(int M, int N, float* mat);

#define TOLERENCE 1E-3
bool verify_matrix(int M, int N, float* mat1, float* mat2);

void debug_print_matrix(int M, int N, float* mat);
void output_matrix_tofile(const char* filename, int M, int N, float* mat);

#endif
