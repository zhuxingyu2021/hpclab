#include "GEMM.h"

void sgemm_naive(int k, int m, int n,
                 float* A, int lda,
                 float* B, int ldb,
                 float* C, int ldc)
{
    int i, j, p;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (p = 0; p < k; p++) {
                C(i, j) += A(i, p)*B(p, j);
            }
        }
    }
}

