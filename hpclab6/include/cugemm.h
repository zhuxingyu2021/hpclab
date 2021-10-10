#ifndef _CUGEMM_H_
#define _CUGEMM_H_

void sgemm_naive(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

void sgemm_fast(int k, int m, int n,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

#endif
