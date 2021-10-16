#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string.h>
#include "cugemm.h"
#include <mpi.h>
#include "cmdline.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define MASTER_PROCESS 0


typedef struct {
    int local_m;
    int local_n;
    int local_k;
}metainfo;



int main(int argc, char** argv) {
    int comm_sz, my_rank;

    float* A_local, * B_local, * C_local;

    float* A = NULL;
    float* B = NULL;
    float* C = NULL;

    int M, N, K;

    int* tmp_cnt_buff = NULL;
    int* tmp_displs_buff = NULL;

    double start, finished;
    int no_single_thread = 0;

    MPI_Datatype MPI_metainfo;
    int metainfo_blocklengths[] = { 1,1,1 };
    MPI_Aint metainfo_displacements[] = { 0,sizeof(int),sizeof(int) * 2 };
    MPI_Datatype metainfo_types[] = { MPI_INT,MPI_INT,MPI_INT };

    metainfo localmnk = { 0,0,0 };
    int localmnk_m, slavemnk_m;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Type_create_struct(3, metainfo_blocklengths, metainfo_displacements, metainfo_types, &MPI_metainfo);
    MPI_Type_commit(&MPI_metainfo);

    // Get cmdline M N K
    cmdline::parser cmdparser;
    cmdparser.add<int>("M",'M',"the M dimension",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("N",'N',"the N dimension",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("K",'K',"the K dimension",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("No-single-thread",'s',"the option to disable comparation to single thread algorithm",
                        false,0,cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    M = cmdparser.get<int>("M");
    N = cmdparser.get<int>("N");
    K = cmdparser.get<int>("K");
    no_single_thread = cmdparser.get<int>("No-single-thread");

    if (my_rank == MASTER_PROCESS) {

        A = (float*)malloc(sizeof(float) * M * K);
        B = (float*)malloc(sizeof(float) * K * N);
        C = (float*)malloc(sizeof(float) * M * N);

        srand((int)time(0));
        random_initalize_matrix(M, K, A);
        random_initalize_matrix(K, N, B);
        memset(C, 0, sizeof(float) * M * N);

        start = MPI_Wtime();

        tmp_cnt_buff = (int*)malloc(sizeof(int) * comm_sz);
        tmp_displs_buff = (int*)malloc(sizeof(int) * comm_sz);
        tmp_displs_buff[0] = 0;

        localmnk.local_n = N;
        localmnk.local_k = K;
        if (M % comm_sz == 0) {
            localmnk_m = M / comm_sz;
            slavemnk_m = M / comm_sz;
        }
        else {
            localmnk_m = M - (M / comm_sz) * (comm_sz - 1);
            slavemnk_m= M / comm_sz;
        }

        tmp_cnt_buff[0] = localmnk_m * K;
        for (int i = 1; i < comm_sz; i++) {
            if (i - 1 == MASTER_PROCESS) tmp_displs_buff[i] = tmp_displs_buff[i - 1] + localmnk_m * K;
            else tmp_displs_buff[i] = tmp_displs_buff[i - 1] + slavemnk_m * K;
            tmp_cnt_buff[i] = slavemnk_m * K;
        }

        localmnk.local_m = slavemnk_m; //localmnk需要被Bcast，所以真实值被localmnk_m暂存
    }

    MPI_Bcast(&localmnk, 1, MPI_metainfo, MASTER_PROCESS, MPI_COMM_WORLD);
    
    if (my_rank == MASTER_PROCESS) {
        localmnk.local_m = localmnk_m;
        A_local = (float*)malloc(sizeof(float) * localmnk.local_m * localmnk.local_k);
        B_local = B;
        C_local = (float*)malloc(sizeof(float) * localmnk.local_m * localmnk.local_n);
    }
    else {
        A_local = (float*)malloc(sizeof(float) * localmnk.local_m * localmnk.local_k);
        B_local = (float*)malloc(sizeof(float) * localmnk.local_k * localmnk.local_n);
        C_local = (float*)malloc(sizeof(float) * localmnk.local_m * localmnk.local_n);
    }

    MPI_Scatterv(A, tmp_cnt_buff, tmp_displs_buff, MPI_FLOAT, A_local, localmnk.local_m * localmnk.local_k, MPI_FLOAT, MASTER_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(B_local, localmnk.local_k * localmnk.local_n, MPI_FLOAT, MASTER_PROCESS, MPI_COMM_WORLD);
    memset(C_local, 0, sizeof(float) * localmnk.local_m * localmnk.local_n);


    int devicecnt;
    cudaGetDeviceCount(&devicecnt);
    int device = my_rank % devicecnt;
    cudaSetDevice(device);
    
    sgemm_fast(localmnk.local_k, localmnk.local_m, localmnk.local_n,
               A_local, localmnk.local_k,
               B_local, localmnk.local_n,
               C_local, localmnk.local_n);

    if (my_rank == MASTER_PROCESS)
    {
        tmp_displs_buff[0] = 0;
        tmp_cnt_buff[0] = localmnk.local_m * N;
        for (int i = 1; i < comm_sz; i++) {
            if (i - 1 == MASTER_PROCESS) tmp_displs_buff[i] = tmp_displs_buff[i - 1] + localmnk.local_m * N;
            else tmp_displs_buff[i] = tmp_displs_buff[i - 1] + slavemnk_m * N;
            tmp_cnt_buff[i] = slavemnk_m * N;

            //cout << i << " " << tmp_cnt_buff[i] << " " << tmp_displs_buff[i] << endl;
        }
    }

    MPI_Gatherv(C_local, localmnk.local_m * localmnk.local_n, MPI_FLOAT, C, tmp_cnt_buff, tmp_displs_buff, MPI_FLOAT, MASTER_PROCESS, MPI_COMM_WORLD);

    if (my_rank == MASTER_PROCESS) {
        finished = MPI_Wtime();

        double tp = finished - start;
        cout << "Time cost by parrallel algorithm: " << tp << "s" << endl;

        if(!no_single_thread) {
            float *C_naive = (float *) malloc(sizeof(float) * M * N);
            memset(C_naive, 0, sizeof(float) * M * N);

            start = MPI_Wtime();
            sgemm_fast(K, M, N, A, K, B, N, C_naive, N);
            finished = MPI_Wtime();

            double ts = finished - start;
            cout << "Time cost by single thread algorithm: " << ts << "s" << endl;
            cout << "Accelerated " << ts / tp << " x" << endl;

            if (!verify_matrix(M, N, C_naive, C)) {
                cerr << "Your optimize method is wrong!" << endl;
            }

            free(C_naive);
        }

        free(A);
        free(B);
        free(C);

        free(tmp_cnt_buff);
        free(tmp_displs_buff);
    }
    else{
        free(B_local);
    }
    free(A_local);
    free(C_local);

    MPI_Type_free(&MPI_metainfo);
    MPI_Finalize();

    return 0;
}