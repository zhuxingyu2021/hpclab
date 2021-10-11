#include <iostream>
#include <pthread.h>
#include "GEMM.h"
#include "cmdline.h"
#include <cstdlib>
#include <time.h>
#include "mytime.h"

using namespace std;

typedef struct{
    float* local_A;
    float* local_B;
    float* local_C;
    int local_M, local_N, local_K;
    int lda, ldb, ldc;
}gemm_worker_arg;

void* gemm_worker(void* args);

int main(int argc, char** argv) {
    cmdline::parser cmdparser;
    cmdparser.add<int>("M",'M',"the M dimension",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("N",'N',"the N dimension",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("K",'K',"the K dimension",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("num_workers",'n',"the number of threads",true,1,cmdline::range(1,16));
    cmdparser.add<int>("No-single-thread",'s',"the option to disable comparation to single thread algorithm",
                        false,0,cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    int n_threads = cmdparser.get<int>("num_workers");
    int no_single_thread = cmdparser.get<int>("No-single-thread");

    int M, N, K, lda, ldb, ldc;
    M = cmdparser.get<int>("M");
    N = cmdparser.get<int>("N");
    K = cmdparser.get<int>("K");
    lda = K;
    ldb = N;
    ldc = N;

    float* A = (float*)aligned_malloc(sizeof(float) * M * K, GEMM_CACHELINE_SIZE);
    float* B = (float*)aligned_malloc(sizeof(float) * K * N, GEMM_CACHELINE_SIZE);
    float* C = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
    float* C_singlethread = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);

    double timestart, timeend;

    srand((int)time(0));
    random_initalize_matrix(M, K, A);
    random_initalize_matrix(K, N, B);
    memset(C, 0, sizeof(float) * M * N);
    memset(C_singlethread, 0, sizeof(float) * M * N);

    timestart = get_wall_time();
    pthread_t threads[n_threads];
    gemm_worker_arg threadarg[n_threads];
    int worker_rows_common = M / n_threads;
    for(int i = 0; i < n_threads; i++){
        int worker_rows = worker_rows_common;
        threadarg[i].local_A = &A(i * worker_rows_common, 0);
        threadarg[i].local_B = B;
        threadarg[i].local_C = &C(i * worker_rows_common, 0);
        threadarg[i].lda = lda;
        threadarg[i].ldb = ldb;
        threadarg[i].ldc = ldc;

        if(i == n_threads - 1) worker_rows = M - worker_rows_common * (n_threads - 1);
        threadarg[i].local_M = worker_rows;
        threadarg[i].local_N = N;
        threadarg[i].local_K = K;

        pthread_create(&threads[i], NULL, gemm_worker, (void*)&threadarg[i]);
    }
    for(int i = 0; i < n_threads; i++){
        pthread_join(threads[i], NULL);
    }
    timeend = get_wall_time();
    double parallel_multiply_time = timeend - timestart;
    cout << "Time cost by parallel gemm: " << parallel_multiply_time << "s" << endl;

    if(!no_single_thread){
        timestart = get_wall_time();
        sgemm_fast(K, M, N, A, K, B, N, C_singlethread, N);
        timeend = get_wall_time();
        double singlethread_multiply_time = timeend - timestart;
        cout << "Time cost by single thread gemm: " << singlethread_multiply_time << "s" << endl;
        cout << "Accelerate ratio: " << singlethread_multiply_time/parallel_multiply_time << "x" << endl;

        if (!verify_matrix(M, N, C, C_singlethread)) {
            cerr << "Your optimize method is wrong!" << endl;
        }
    }

    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    aligned_free(C_singlethread);
    return 0;
}

void* gemm_worker(void* args){
    gemm_worker_arg* callarg = (gemm_worker_arg*)args;

    sgemm_fast(callarg->local_K, callarg->local_M, callarg->local_N,
               callarg->local_A, callarg->lda,
               callarg->local_B, callarg->ldb,
               callarg->local_C, callarg->ldc);

    return NULL;
}
