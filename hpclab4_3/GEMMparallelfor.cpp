#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string.h>
#include "GEMM.h"
#include "mytime.h"
#include "cmdline.h"

#ifdef INTEL_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

using namespace std;

// Run gemm by openblas/MKL
void gemm_blas_multiply(int k, int m, int n,
                        float* A, int lda,
                        float* B, int ldb,
                        float* C, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, A, lda, B, ldb, 0, C, ldc);
}

int main(int argc, char** argv) {
    cmdline::parser cmdparser;
    cmdparser.add<int>("M",'M',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("N",'N',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("K",'K',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("Multiple-runs",'m',"enable multiple runs",
                       false,0,cmdline::oneof(0, 1));
    cmdparser.add<double>("Time-limit",'t',"multiple runs time limit",
                          false,0,cmdline::range(0, 3600));
    cmdparser.add<int>("Easy",'e',"easier cmdline output",
                       false,0,cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    int M = cmdparser.get<int>("M");
    int N = cmdparser.get<int>("N");
    int K = cmdparser.get<int>("K");
    int multiple_runs = cmdparser.get<int>("Multiple-runs");
    double time_limit = 0.0;
    if(multiple_runs) time_limit = cmdparser.get<double>("Time-limit");
    int easy_cmd = cmdparser.get<int>("Easy");

#ifdef INTEL_MKL
    float* A = (float*)MKL_malloc(sizeof(float) * M * K,GEMM_CACHELINE_SIZE);
    float* B = (float*)MKL_malloc(sizeof(float) * K * N, GEMM_CACHELINE_SIZE);
    float* C_blas = (float*)MKL_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    float* C_gemm = (float*)MKL_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
#else
    float* A = (float*)aligned_malloc(sizeof(float) * M * K,GEMM_CACHELINE_SIZE);
    float* B = (float*)aligned_malloc(sizeof(float) * K * N, GEMM_CACHELINE_SIZE);
    float* C_blas = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    float* C_gemm = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
#endif

    double timestart;
    double timeend;

    srand((int)time(0));
    random_initalize_matrix(M, K, A);
    random_initalize_matrix(K, N, B);
    memset(C_blas, 0, sizeof(float) * M * N);
    memset(C_gemm, 0, sizeof(float) * M * N);

    timestart = get_wall_time();
    double blas_multiply_time = timeend - timestart;
    int r_times = 0;
    //sgemm_naive(K, M, N, A, K, B, N, C_naive, N);
    while(1){
        r_times++;
        gemm_blas_multiply(K, M, N, A, K, B, N, C_blas, N);
        timeend = get_wall_time();
        blas_multiply_time = timeend - timestart;
        if(blas_multiply_time > time_limit) break;
    }
#ifdef INTEL_MKL
    if(!easy_cmd){
        cout << "Time cost by mkl gemm: " << blas_multiply_time/r_times << "s" << endl;
    }
    else{
        cout << M << " " << N << " " << K << endl;
        cout << blas_multiply_time/r_times << endl;
    }
#else
    cout << "Time cost by openblas gemm: " << blas_multiply_time/r_times << "s" << endl;
#endif

    timestart = get_wall_time();
    double optimized_multiply_time = timeend - timestart;
    r_times = 0;
    while(1){
        r_times++;
        sgemm_fast(K, M, N, A, K, B, N, C_gemm, N);
        timeend = get_wall_time();
        optimized_multiply_time = timeend - timestart;
        if(optimized_multiply_time > time_limit) break;
    }
    if(!easy_cmd){
        cout << "Time cost by optimized gemm: " << optimized_multiply_time/r_times << "s" << endl;
    }
    else{
        cout << optimized_multiply_time/r_times << endl;
    }

    if (r_times == 1){
        if (!verify_matrix(M, N, C_blas, C_gemm)) {
            cerr << "Your optimize method is wrong!" << endl;
            //output_matrix_tofile("A.csv", M, K, A);
            //output_matrix_tofile("B.csv", K, N, B);
            //output_matrix_tofile("cnaive.csv", M, N, C_naive);
            //output_matrix_tofile("cgemm.csv", M, N, C_gemm);
        }
    }


    float* C_naive = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    memset(C_naive, 0, sizeof(float) * M * N);

    timestart = get_wall_time();
    double naive_multiply_time = timeend - timestart;
    r_times = 0;
    while(1){
        r_times++;
        sgemm_naive(K, M, N, A, K, B, N, C_naive, N);
        timeend = get_wall_time();
        naive_multiply_time = timeend - timestart;
        if(naive_multiply_time > time_limit) break;
    }
    if(!easy_cmd){
        cout << "Time cost by naive gemm: " << naive_multiply_time/r_times << "s" << endl;
    }
    else{
        cout << naive_multiply_time/r_times << endl;
    }

    aligned_free(C_naive);
#ifdef INTEL_MKL
    MKL_free(A);
    MKL_free(B);
    MKL_free(C_blas);
    MKL_free(C_gemm);
#else
    aligned_free(A);
    aligned_free(B);
    aligned_free(C_blas);
    aligned_free(C_gemm);
#endif
}
