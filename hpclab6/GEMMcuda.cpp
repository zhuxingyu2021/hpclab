#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>

#include "mytime.h"
#include "cugemm.h"
#include "cmdline.h"

using namespace std;

// Run gemm by cublas
void sgemm_blas(int k, int m, int n,
    float* A, int lda,
    float* B, int ldb,
    float* C, int ldc)
{
    float* d_A, * d_B, * d_C;
    float alpha = 1.0;
    float beta = 0;
    cudaMalloc(&d_A, sizeof(float) * m * k);
    cudaMalloc(&d_B, sizeof(float) * k * n);
    cudaMalloc(&d_C, sizeof(float) * m * n);
    cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(float) * m * n);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublas按照column-major方式存储，所以要调用时要调换一些参数的位置
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, d_B, n, d_A, k, &beta, d_C, n);

    cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    cmdline::parser cmdparser;
    cmdparser.add<int>("M", 'M', "the M dimension", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("N", 'N', "the N dimension", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("K", 'K', "the K dimension", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("Multiple-runs", 'm', "enable multiple runs",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.add<double>("Time-limit", 't', "multiple runs time limit",
        false, 0, cmdline::range(0, 3600));
    cmdparser.add<int>("Easy", 'e', "easier cmdline output",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    int M = cmdparser.get<int>("M");
    int N = cmdparser.get<int>("N");
    int K = cmdparser.get<int>("K");
    int multiple_runs = cmdparser.get<int>("Multiple-runs");
    double time_limit = 0.0;
    if (multiple_runs) time_limit = cmdparser.get<double>("Time-limit");
    int easy_cmd = cmdparser.get<int>("Easy");


    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * K * N);
    float* C_blas = (float*)malloc(sizeof(float) * M * N);
    float* C_gemm = (float*)malloc(sizeof(float) * M * N);

    double timestart;
    double timeend;

    srand((int)time(0));
    random_initalize_matrix(M, K, A);
    random_initalize_matrix(K, N, B);

    timestart = get_wall_time();
    double blas_multiply_time;
    sgemm_blas(K, M, N, A, K, B, N, C_blas, N);
    timeend = get_wall_time();
    blas_multiply_time = timeend - timestart;

    if (!easy_cmd) {
        cout << "Time cost by cublas gemm: " << blas_multiply_time / 1 << "s" << endl;
    }
    else {
        cout << M << " " << N << " " << K << endl;
        cout << blas_multiply_time / 1 << endl;
    }

    timestart = get_wall_time();
    double optimized_multiply_time;
    sgemm_fast(K, M, N, A, K, B, N, C_gemm, N);
    timeend = get_wall_time();
    optimized_multiply_time = timeend - timestart;

    if (!easy_cmd) {
        cout << "Time cost by optimized gemm: " << optimized_multiply_time / 1 << "s" << endl;
    }
    else {
        cout << M << " " << N << " " << K << endl;
        cout << optimized_multiply_time / 1 << endl;
    }

    if (!verify_matrix(M, N, C_gemm, C_blas)) {
        cerr << "Your optimize method is wrong!" << endl;
        //output_matrix_tofile("A.csv", M, K, A);
        //output_matrix_tofile("B.csv", K, N, B);
        //output_matrix_tofile("C_blas.csv", M, N, C_blas);
        //output_matrix_tofile("C_gemm.csv", M, N, C_gemm);
    }

    free(A);
    free(B);
    free(C_blas);
    free(C_gemm);
}
