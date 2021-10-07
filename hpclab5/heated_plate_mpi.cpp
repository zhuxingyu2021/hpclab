# include <stdio.h>
# include <math.h>
# include <stdlib.h>
# include "common_define.h"
# include <mpi.h>
# include <assert.h>

# define MASTER_PROCESS 0
# define local_u(i, j) local_u[(i) * N + (j)]
# define local_w(i, j) local_w[(i) * N + (j)]

# define MAX(i, j) (((i) > (j))?(i):(j))
# define MIN(i, j) (((i) < (j))?(i):(j))

int main ( int argc, char *argv[])
{
    int my_rank, comm_sz;

    double diff;
    double epsilon = 0.001;
    int iterations;
    int iterations_print;
    double mean;
    double my_diff;
    double wtime;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int prev_rank = my_rank - 1;
    if(prev_rank < 0) prev_rank = MPI_PROC_NULL;
    int next_rank = my_rank + 1;
    if(next_rank >= comm_sz) next_rank = MPI_PROC_NULL;

    if(my_rank == MASTER_PROCESS) {
        printf("\n");
        printf("HEATED_PLATE_MPI\n");
        printf("  MPI version\n");
        printf("  A program to solve for the steady state temperature distribution\n");
        printf("  over a rectangular plate.\n");
        printf("\n");
        printf("  Spatial grid of %d by %d points.\n", M, N);
        printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
        printf("  Number of threads =              %d\n", comm_sz);
    }

    int local_m = M/comm_sz;
    int offset_m = local_m * my_rank;
    if(my_rank == comm_sz - 1)
    {
        local_m = M - local_m * (comm_sz - 1);
    }
    int offset_end_m = offset_m + local_m;


    double* local_u = (double *) malloc(sizeof(double) * local_m * N);
    double* local_w = (double *) malloc(sizeof(double) * local_m * N);

/*
  Set the boundary values, which don't change.
*/

    mean = 0.0;

    for(int i = MAX(0, 1 - offset_m); i < MIN(local_m, M - 1 - offset_m); i++)
    {
        local_w(i,0) = 100.0;
        local_w(i,N-1) = 100.0;
    }
    if(M - 1 < offset_end_m){
        for (int j = 0; j < N; j++)
        {
            local_w(M - 1 - offset_m, j) = 100.0;
        }
    }
    if(offset_m == 0){
        for (int j = 0; j < N; j++)
        {
            local_w(0, j) = 0.0;
        }
    }

/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
    double local_sum = 0.0;

    for(int i = MAX(0, 1 - offset_m); i < MIN(local_m, M - 1 - offset_m); i++) {
        local_sum = local_sum + local_w(i, 0) + local_w(i, N - 1);
    }
    if(M - 1 < offset_end_m){
        for (int j = 0; j < N; j++)
        {
            local_sum += local_w(M - 1 - offset_m, j);
        }
    }
    if(offset_m == 0){
        for (int j = 0; j < N; j++)
        {
            local_sum += local_w(0, j);
        }
    }
    MPI_Allreduce(&local_sum, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    mean = mean / ( double ) ( 2 * M + 2 * N - 4 );

    if(my_rank == MASTER_PROCESS) {
        printf("\n");
        printf("  MEAN = %f\n", mean);
    }

/*
  Initialize the interior solution to the mean value.
*/
    for(int i = MAX(0, 1 - offset_m); i < MIN(local_m, M - 1 - offset_m); i++) {
        for (int j = 1; j < N - 1; j++ )
        {
            local_w(i, j) = mean;
        }
    }

/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
    iterations = 0;
    iterations_print = 1;
    if(my_rank == MASTER_PROCESS) {
        printf("\n");
        printf(" Iteration  Change\n");
        printf("\n");
        wtime = MPI_Wtime();
    }

    double recvbuf_prev[N];
    double recvbuf_next[N];
    MPI_Request req_s[2];
    MPI_Status sta_s[2];
    MPI_Request req_r[2];
    MPI_Status sta_r[2];

    diff = epsilon;

    while(epsilon <= diff)
    {
/*
  Save the old solution in U.
*/
        for(int i = 0; i < local_m; i++)
        {
            for (int j = 0; j < N; j++ )
            {
                local_u(i, j) = local_w(i, j);
            }
        }

/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/
        MPI_Isend(local_u, N, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &req_s[0]);
        MPI_Isend(&local_u(local_m-1, 0), N, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &req_s[1]);
        //MPI_Send(local_u, N, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD);
        //MPI_Recv(recvbuf_next, N, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Send(&local_u(local_m-1, 0), N, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
        //MPI_Recv(recvbuf_prev, N, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 1; i < local_m - 1; i++) {
            for (int j = 1; j < N - 1; j++ )
            {
                local_w(i, j) = ( local_u(i-1,j) + local_u(i+1,j) + local_u(i,j-1) + local_u(i,j+1) ) / 4.0;
            }
        }

        MPI_Irecv(recvbuf_prev, N, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &req_r[0]);
        MPI_Irecv(recvbuf_next, N, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &req_r[1]);

        MPI_Wait(&req_s[0], &sta_s[0]);
        MPI_Wait(&req_s[1], &sta_s[1]);
        MPI_Wait(&req_r[0], &sta_r[0]);
        MPI_Wait(&req_r[1], &sta_r[1]);

        if(offset_m > 0)
        {
            for (int j = 1; j < N - 1; j++ ) {
                local_w(0, j) = (recvbuf_prev[j] + local_u(1, j) + local_u(0, j - 1) + local_u(0, j + 1)) / 4.0;
            }
        }
        if(M - 1 >= offset_end_m)
        {
            for (int j = 1; j < N - 1; j++){
                local_w(local_m-1, j) = ( local_u(local_m-2,j) + recvbuf_next[j] +
                        local_u(local_m-1,j-1) + local_u(local_m-1,j+1) ) / 4.0;
            }
        }

        my_diff = 0.0;
        for(int i = MAX(0, 1 - offset_m); i < MIN(local_m, M - 1 - offset_m); i++) {
            for (int j = 1; j < N - 1; j++ )
            {
                if ( my_diff < fabs ( local_w(i, j) - local_u(i, j) ) )
                {
                    my_diff = fabs ( local_w(i, j) - local_u(i, j) );
                }
            }
        }

        MPI_Allreduce(&my_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        iterations++;
        if ( iterations == iterations_print && my_rank == MASTER_PROCESS)
        {
            printf ( "  %8d  %f\n", iterations, diff );
            iterations_print = 2 * iterations_print;
        }
    }

    if(my_rank == MASTER_PROCESS) {
        wtime = MPI_Wtime() - wtime;
        printf("\n");
        printf("  %8d  %f\n", iterations, diff);
        printf("\n");
        printf("  Error tolerance achieved.\n");
        printf("  Wallclock time = %f\n", wtime);
/*
  Terminate.
*/
        printf("\n");
        printf("HEATED_PLATE_MPI:\n");
        printf("  Normal end of execution.\n");
    }

    free(local_w);
    free(local_u);
    MPI_Finalize();
    return 0;
}
