#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string.h>
#include "GEMM.h"
#include <mpi.h>

#include <fstream>
#include "cmdline.h"

using namespace std;

#define MASTER_PROCESS 0


typedef struct {
    int local_m;
    int local_n;
    int local_k;
}metainfo;


int main(int argc, char** argv) {
    int comm_sz, my_rank;
    int comm_rows, comm_cols;

    float *A_local, *B_local, *C_local;

    float* A = NULL;
    float* B = NULL;
    float* C = NULL;

    int M, N, K;

    double start, finished;

    int no_single_thread = 0;

    MPI_Datatype MPI_metainfo;
    int metainfo_blocklengths[] = { 1,1,1 };
    MPI_Aint metainfo_displacements[] = { 0,sizeof(int),sizeof(int) * 2 };
    MPI_Datatype metainfo_types[] = { MPI_INT,MPI_INT,MPI_INT };

    metainfo localmnk = { 0,0,0 };
    metainfo slavemnk = { 0,0,0 };

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Type_create_struct(3, metainfo_blocklengths, metainfo_displacements, metainfo_types, &MPI_metainfo);
    MPI_Type_commit(&MPI_metainfo);

    // Get cmdline M N K
    cmdline::parser cmdparser;
    cmdparser.add<int>("M",'M',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("N",'N',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("K",'K',"the number of threads",true,512,cmdline::range(1,65536));
    cmdparser.add<int>("No-single-thread",'s',"the option to disable comparation to single thread algorithm",
                       false,0,cmdline::oneof(0, 1));
    cmdparser.parse_check(argc, argv);

    M = cmdparser.get<int>("M");
    N = cmdparser.get<int>("N");
    K = cmdparser.get<int>("K");
    no_single_thread = cmdparser.get<int>("No-single-thread");

    if (my_rank == MASTER_PROCESS) {
        float* p_A;

        A = (float*)aligned_malloc(sizeof(float) * M * K, GEMM_CACHELINE_SIZE);
        B = (float*)aligned_malloc(sizeof(float) * K * N, GEMM_CACHELINE_SIZE);
        C = (float*)aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);

        srand((int)time(0));
        random_initalize_matrix(M, K, A);
        random_initalize_matrix(K, N, B);
        memset(C, 0, sizeof(float) * M * N);

        start = MPI_Wtime();

        localmnk.local_n = N;
        localmnk.local_k = K;
        slavemnk.local_n = N;
        slavemnk.local_k = K;
        if (M % comm_sz == 0) {
            localmnk.local_m = M / comm_sz;
            slavemnk.local_m = M / comm_sz;
        }
        else {
            localmnk.local_m = M - (M / comm_sz) * (comm_sz - 1);
            slavemnk.local_m = M / comm_sz;
        }
        //cout << localmnk.local_m << " " << slavemnk.local_m << endl;

        p_A = A + localmnk.local_m * K;

        for (int i = 1; i < comm_sz; i++) {
            MPI_Send(&slavemnk, 1, MPI_metainfo, i, 0, MPI_COMM_WORLD);
            MPI_Send(p_A, slavemnk.local_m * K, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(B, N * K, MPI_FLOAT, i, 0, MPI_COMM_WORLD);

            p_A += slavemnk.local_m * K;
        }

        A_local = A;
        B_local = B;
        C_local = C;

        //output_matrix_tofile("A.csv", M, K, A);
        //output_matrix_tofile("B.csv", K, N, B);
    }
    else {
        MPI_Recv(&localmnk, 1, MPI_metainfo, MASTER_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        A_local = (float*)aligned_malloc(sizeof(float) * localmnk.local_m * localmnk.local_k, GEMM_CACHELINE_SIZE);
        B_local = (float*)aligned_malloc(sizeof(float) * localmnk.local_k * localmnk.local_n, GEMM_CACHELINE_SIZE);
        C_local = (float*)aligned_malloc(sizeof(float) * localmnk.local_m * localmnk.local_n, GEMM_CACHELINE_SIZE);

        MPI_Recv(A_local, localmnk.local_m * localmnk.local_k, MPI_FLOAT, MASTER_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_local, localmnk.local_k * localmnk.local_n, MPI_FLOAT, MASTER_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memset(C_local, 0, sizeof(float) * localmnk.local_m * localmnk.local_n);
    }

    //string filenameA = to_string(my_rank) + string("A.csv");
    //output_matrix_tofile(filenameA.c_str(), localmnk.local_m, localmnk.local_k, A_local);

    //print_in_sync(my_rank, comm_sz, MASTER_PROCESS, localmnk.local_m, localmnk.local_n, localmnk.local_k);

    gemm_fast_multiply(localmnk.local_k, localmnk.local_m, localmnk.local_n,
        A_local, localmnk.local_k,
        B_local, localmnk.local_n,
        C_local, localmnk.local_n);

    //string filenameA = to_string(my_rank) + string("C.csv");
    //output_matrix_tofile(filenameA.c_str(), localmnk.local_m, localmnk.local_n, C_local);

    if (my_rank == MASTER_PROCESS)
    {
        float* p_C = C + localmnk.local_m * N;

        for (int i = 1; i < comm_sz; i++) {
            MPI_Recv(p_C, slavemnk.local_m * slavemnk.local_n, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            p_C += slavemnk.local_m * slavemnk.local_n;
        }

        finished = MPI_Wtime();

        double tp = finished - start;
        cout << "Time cost by parrallel algorithm: " << tp << "s" << endl;


        if(!no_single_thread) {
            float *C_naive = (float *) aligned_malloc(sizeof(float) * M * N, GEMM_CACHELINE_SIZE);
            memset(C_naive, 0, sizeof(float) * M * N);

            start = MPI_Wtime();
            gemm_fast_multiply(K, M, N, A, K, B, N, C_naive, N);
            finished = MPI_Wtime();

            double ts = finished - start;
            cout << "Time cost by single thread algorithm: " << ts << "s" << endl;
            cout << "Accelerated " << ts / tp << " x" << endl;

            if (!verify_matrix(M, N, C_naive, C)) {
                cerr << "Your optimize method is wrong!" << endl;
            }
            aligned_free(C_naive);
        }

        aligned_free(A);
        aligned_free(B);
        aligned_free(C);

    }
    else {
        MPI_Send(C_local, localmnk.local_m * localmnk.local_n, MPI_FLOAT, MASTER_PROCESS, 0, MPI_COMM_WORLD);
        aligned_free(A_local);
        aligned_free(B_local);
        aligned_free(C_local);
    }

    MPI_Type_free(&MPI_metainfo);
    MPI_Finalize();

    return 0;
}
