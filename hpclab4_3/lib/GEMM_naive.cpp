#include "GEMM.h"
#include "parallelfor.h"
#include <iostream>

typedef struct{
    int k; int m; int n;
    float* A; int lda;
    float* B; int ldb;
    float* C; int ldc;
}sgemm_naive_single_thread_kernel_args;
void* sgemm_naive_single_thread_kernel(void* pf_args) {
    // 读取传入参数
    int k = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->k;
    int m = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->m;
    int n = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->n;
    int lda = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->lda;
    int ldb = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->ldb;
    int ldc = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->ldc;
    float* A = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->A;
    float* B = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->B;
    float* C = PF_GET_PARG(pf_args, sgemm_naive_single_thread_kernel_args)->C;

    int i;
    for (i = PF_GET_INDEX_START(pf_args); i < PF_GET_INDEX_END(pf_args); i += PF_GET_INDEX_INCREMENT(pf_args)) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }

    return NULL;
}

void sgemm_naive(int k, int m, int n,
                 float* A, int lda,
                 float* B, int ldb,
                 float* C, int ldc, int n_threads)
{
    // 准备子线程传入参数
    sgemm_naive_single_thread_kernel_args arg;
    arg.k = k; arg.m = m; arg.n = n;
    arg.A = A; arg.lda = lda;
    arg.B = B; arg.ldb = ldb;
    arg.C = C; arg.ldc = ldc;

    parallel_for(0, m, 1, &sgemm_naive_single_thread_kernel, (void*)&arg, n_threads);
}

