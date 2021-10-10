#include "GEMM.h"
#include "common_define.h"
#include <omp.h>
#include <iostream>

void sgemm_naive(int k, int m, int n,
                 float* A, int lda,
                 float* B, int ldb,
                 float* C, int ldc)
{
    int i;

#pragma omp parallel for num_threads(N_THREADS) private(i)
        for (i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int p = 0; p < k; p++) {
                    C(i, j) += A(i, p) * B(p, j);
                }
            }
        }
}

