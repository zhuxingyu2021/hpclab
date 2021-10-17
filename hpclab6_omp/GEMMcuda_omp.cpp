#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include "cugemm.h"
#include "cmdline.h"
#include "mytime.h"

using namespace std;

int main(int argc, char** argv) {
    cmdline::parser cmdparser;
    cmdparser.add<int>("M", 'M', "the M dimension", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("N", 'N', "the N dimension", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("K", 'K', "the K dimension", true, 512, cmdline::range(1, 65536));
    cmdparser.add<int>("num_workers", 'n', "the number of threads", true, 1, cmdline::range(1, 16));
    cmdparser.add<int>("Multiple-runs", 'm', "enable multiple runs",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.add<int>("run-times", 't', "kernel run times",
        false, 1, cmdline::range(0, 3600));
    cmdparser.add<int>("Easy", 'e', "easier cmdline output",
        false, 0, cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    int M = cmdparser.get<int>("M");
    int N = cmdparser.get<int>("N");
    int K = cmdparser.get<int>("K");
    int multiple_runs = cmdparser.get<int>("Multiple-runs");
    int run_times = 1;
    if (multiple_runs) run_times = cmdparser.get<int>("run-times");
    int easy_cmd = cmdparser.get<int>("Easy");
    int n_threads = cmdparser.get<int>("num_workers");

    float *A, *B, *C_singlethread, *C_multithread;
    //·ÖÅäÒ³Ëø¶¨ÄÚ´æ
    checkCudaErrors(cudaHostAlloc(&A, sizeof(float) * M * K, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&B, sizeof(float) * K * N, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&C_singlethread, sizeof(float) * M * N, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&C_multithread, sizeof(float) * M * N, cudaHostAllocDefault));

    srand((int)time(0));
    random_initalize_matrix(M, K, A);
    random_initalize_matrix(K, N, B);
    //output_matrix_tofile("A.csv", M, K, A);

    //debug_print_matrix(M, K, A);
    //cout << endl;
    //debug_print_matrix(K, N, B);

    double timestart = .0, timeend = .0;

    int t = run_times;
    double singlethread_multiply_time = .0;
    while (t > 0)
    {
        singlethread_multiply_time += sgemm_fast(K, M, N, A, K, B, N, C_singlethread, N);
        t--;
    }

    if (!easy_cmd) {
        cout << "Time cost by cublas gemm: " << singlethread_multiply_time / run_times << "s" << endl;
    }
    else {
        cout << M << " " << N << " " << K << endl;
        cout << (timeend - timestart) / run_times << endl;
    }

    t = run_times;
    double multithread_multiply_time = .0;
    while (t > 0) {
        multithread_multiply_time += sgemm_fast_multithread(K, M, N, A, K, B, N, C_multithread, N, n_threads);
        t--;
    }

    if (!easy_cmd) {
        cout << "Time cost by optimized gemm: " << multithread_multiply_time / run_times << "s" << endl;
    }
    else {
        cout << (timeend - timestart) / run_times << endl;
    }

    if (run_times == 1 && (!verify_matrix(M, N, C_multithread, C_singlethread))) {
        cerr << "Your optimize method is wrong!" << endl;
        //output_matrix_tofile("A.csv", K, N, A);
        //output_matrix_tofile("B.csv", K, N, B);
        //output_matrix_tofile("C_multithread.csv", M, N, C_multithread);
        //output_matrix_tofile("C_singlethread.csv", M, N, C_singlethread);
    }

    checkCudaErrors(cudaFreeHost(A));
    checkCudaErrors(cudaFreeHost(B));
    checkCudaErrors(cudaFreeHost(C_multithread));
    checkCudaErrors(cudaFreeHost(C_singlethread));
}
