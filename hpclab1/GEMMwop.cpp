#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string.h>
#include <cblas.h>
#include "GEMM.h"
#include "cmdline.h"

using namespace std;

// Run gemm by openblas
void gemm_blas_multiply(int k, int m, int n,
                         float* A, int lda,
                         float* B, int ldb,
                         float* C, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, A, lda, B, ldb, 0, C, ldc);
}

int main(int argc, char** argv) {
    int M, N, K;

    cmdline::parser cmdparser;
    cmdparser.add<int>("M",'M',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("N",'N',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("K",'K',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.parse_check(argc, argv);

    M = cmdparser.get<int>("M");
    N = cmdparser.get<int>("N");
    K = cmdparser.get<int>("K");

    cout << "M = " << M << ", N = " << N << ", K = " << K << endl;

    float* A = (float*)aligned_malloc(sizeof(float) * M * K,GEMM_CACHELINE_SIZE);
    float* B = (float*)aligned_malloc(sizeof(float) * K * N, GEMM_CACHELINE_SIZE);
    float* C_naive = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    float* C_gemm = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    float* C_blas = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);

    int timestart;
    int timeend;

    srand((int)time(0));
    random_initalize_matrix(M, K, A);
    random_initalize_matrix(K, N, B);
    memset(C_naive, 0, sizeof(float) * M * N);
    memset(C_gemm, 0, sizeof(float) * M * N);
    memset(C_blas, 0, sizeof(float) * M * N);

    timestart = clock();
    gemm_blas_multiply(K, M, N, A, K, B, N, C_blas, N);
    timeend = clock();
    double blas_multiply_time = double(timeend - timestart) / CLOCKS_PER_SEC;
    cout << "Time cost by openblas: " << blas_multiply_time << "s" << endl;

    timestart = clock();
    gemm_fast_multiply(K, M, N, A, K, B, N, C_gemm, N);
    timeend = clock();
    double optimized_multiply_time = double(timeend - timestart) / CLOCKS_PER_SEC;
    cout << "Time cost by optimized gemm: " << optimized_multiply_time << "s" << endl;

    timestart = clock();
    gemm_naive_multiply(K, M, N, A, K, B, N, C_naive, N);
    timeend = clock();
    double naive_multiply_time = double(timeend - timestart) / CLOCKS_PER_SEC;
    cout << "Time cost by naive gemm: " << naive_multiply_time << "s" << endl;
    cout << "Accelerate ratio: " << naive_multiply_time/optimized_multiply_time << "x" << endl;

    if (!verify_matrix(M, N, C_blas, C_gemm)) {
        cerr << "Your optimize method is wrong!" << endl;
        //output_matrix_tofile("A.csv", M, K, A);
        //output_matrix_tofile("B.csv", K, N, B);
        //output_matrix_tofile("cnaive.csv", M, N, C_naive);
        //output_matrix_tofile("cgemm.csv", M, N, C_gemm);
    }

    aligned_free(A);
    aligned_free(B);
    aligned_free(C_naive);
    aligned_free(C_gemm);
    aligned_free(C_blas);
}
