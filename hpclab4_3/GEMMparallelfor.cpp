#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string.h>
#include "GEMM.h"
#include <cmath>
#include "mythreads_utils.h"

using namespace std;

int main() {
    int M, N, K;
    cout << "Input M,N,K:(512-2048)" << endl;
    cin >> M >> N >> K;

    float* A = (float*)aligned_malloc(sizeof(float) * M * K,GEMM_CACHELINE_SIZE);
    float* B = (float*)aligned_malloc(sizeof(float) * K * N, GEMM_CACHELINE_SIZE);
    float* C_naive = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    float* C_gemm = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);

    double timestart;
    double timeend;

    srand((int)time(0));
    random_initalize_matrix(M, K, A);
    random_initalize_matrix(K, N, B);
    memset(C_naive, 0, sizeof(float) * M * N);
    memset(C_gemm, 0, sizeof(float) * M * N);

    timestart = get_wall_time();
    gemm_naive_multiply(K, M, N, A, K, B, N, C_naive, N);
    timeend = get_wall_time();
    double naive_multiply_time = timeend - timestart;
    cout << "Time cost by naive gemm: " << naive_multiply_time << "s" << endl;

    timestart = get_wall_time();
    gemm_fast_multiply(K, M, N, A, K, B, N, C_gemm, N);
    timeend = get_wall_time();
    double optimized_multiply_time = timeend - timestart;
    cout << "Time cost by optimized gemm: " << optimized_multiply_time << "s" << endl;
    cout << "Accelerate ratio: " << naive_multiply_time/optimized_multiply_time << "x" << endl;

    if (!verify_matrix(M, N, C_naive, C_gemm)) {
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
}
