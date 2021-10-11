# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include "parallelfor.h"
# include "mytime.h"
# include <unistd.h>
#include "common_define.h"

double u[M][N];
double w[M][N];

int main ( int argc, char *argv[] );

/******************************************************************************/

int main ( int argc, char *argv[] )
{
    double diff;
    double epsilon = 0.001;
    int iterations;
    int iterations_print;
    double mean;
    double wtime;

    printf ( "\n" );
    printf ( "HEATED_PLATE_PARALLEL_FOR\n" );
    printf ( "  Parallel For version\n" );
    printf ( "  A program to solve for the steady state temperature distribution\n" );
    printf ( "  over a rectangular plate.\n" );
    printf ( "\n" );
    printf ( "  Spatial grid of %d by %d points.\n", M, N );
    printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon );
    printf ("  Number of threads =              %d\n", N_THREADS );

    pf_thread_pool_t* thread_pool = pf_create_thread_pool(N_THREADS);
/*
  Set the boundary values, which don't change. 
*/
    mean = 0.0;

      /*
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][0] = 100.0;
    }
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][N-1] = 100.0;
    }
       */
    parallel_for_cyclic(1, M - 1, 1, [](void* pf_args) -> void* {
        double (*_w)[N] = PF_GET_ARG(pf_args, double(*)[N]);
        PF_FOR_LOOP(i, pf_args)
        {
          _w[i][0] = 100.0;
          _w[i][N-1] = 100.0;
        }
      return NULL;
      }, (void*) w, thread_pool, CYCLIC_SIZE);

      /*
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[M-1][j] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[0][j] = 0.0;
    }*/
    parallel_for_cyclic(0, N, 1, [](void* pf_args) -> void* {
        double (*_w)[N] = PF_GET_ARG(pf_args, double(*)[N]);
        PF_FOR_LOOP(j, pf_args)
        {
          _w[M-1][j] = 100.0;
          _w[0][j] = 0.0;
        }
      return NULL;
    }, (void*) w, thread_pool, CYCLIC_SIZE);

    pf_barrier(thread_pool);

/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.

#pragma omp for reduction ( + : mean )
    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + w[i][0] + w[i][N-1];
    }
#pragma omp for reduction ( + : mean )
    for ( j = 0; j < N; j++ )
    {
      mean = mean + w[M-1][j] + w[0][j];
    }
  */
    typedef struct{
        double (*w)[N];
        double mean;
    }w_mean_arg_t;
    w_mean_arg_t w_mean_arg = {w, mean};
    parallel_for_cyclic(1, M - 1, 1, [](void* pf_args) -> void* {
        double _mean = 0.0;
        double (*_w)[N] = PF_GET_PARG(pf_args, w_mean_arg_t) -> w;
        PF_FOR_LOOP(i, pf_args)
        {
            _mean = _mean + _w[i][0] + _w[i][N-1];
        }
        PF_CRITIAL_BEGIN(pf_args)
        PF_GET_PARG(pf_args, w_mean_arg_t) -> mean += _mean;
        PF_CRITIAL_END(pf_args)
        return NULL;
    }, (void*) &w_mean_arg, thread_pool, CYCLIC_SIZE);
    parallel_for_cyclic(0, N, 1, [](void* pf_args) -> void* {
        double _mean = 0.0;
        double (*_w)[N] = PF_GET_PARG(pf_args, w_mean_arg_t) -> w;
        PF_FOR_LOOP(j, pf_args)
        {
            _mean = _mean + _w[M-1][j] + _w[0][j];
        }
        PF_CRITIAL_BEGIN(pf_args)
        PF_GET_PARG(pf_args, w_mean_arg_t) -> mean += _mean;
        PF_CRITIAL_END(pf_args)
        return NULL;
    }, (void*) &w_mean_arg, thread_pool, CYCLIC_SIZE);

    pf_barrier_with_master(thread_pool);

    mean = w_mean_arg.mean;

    mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
    printf ( "\n" );
    printf ( "  MEAN = %f\n", mean );

/* 
  Initialize the interior solution to the mean value.
*/

/*
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      for ( j = 1; j < N - 1; j++ )
      {
        w[i][j] = mean;
      }
    }
  */
    w_mean_arg.mean = mean;
    parallel_for_cyclic(1, M - 1, 1, [](void* pf_args) -> void*{
        double (*_w)[N] = PF_GET_PARG(pf_args, w_mean_arg_t) -> w;
        double _mean = PF_GET_PARG(pf_args, w_mean_arg_t) -> mean;
        PF_FOR_LOOP(i, pf_args)
        {
            for (int j = 1; j < N - 1; j++ )
            {
                _w[i][j] = _mean;
            }
        }
        return NULL;
    }, (void*)&w_mean_arg, thread_pool, CYCLIC_SIZE);
    pf_barrier_with_master(thread_pool);
/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
    iterations = 0;
    iterations_print = 1;
    printf ( "\n" );
    printf ( " Iteration  Change\n" );
    printf ( "\n" );
    wtime = get_wall_time();

    diff = epsilon;

    while ( epsilon <= diff )
    {
/*
  Save the old solution in U.

# pragma omp for
      for ( i = 0; i < M; i++ ) 
      {
        for ( j = 0; j < N; j++ )
        {
          u[i][j] = w[i][j];
        }
      }
*/
        typedef struct {
            double (*w)[N];
            double (*u)[N];
        }w_u_arg_t;
        w_u_arg_t w_u_arg = {w, u};
        parallel_for_cyclic(0, M, 1, [](void* pf_args) -> void*{
            double (*_w)[N] = PF_GET_PARG(pf_args, w_u_arg_t) -> w;
            double (*_u)[N] = PF_GET_PARG(pf_args, w_u_arg_t) -> u;
            PF_FOR_LOOP(i, pf_args)
            {
                for (int j = 0; j < N; j++ )
                {
                    _u[i][j] = _w[i][j];
                }
            }
            return NULL;
        }, (void*)&w_u_arg, thread_pool, CYCLIC_SIZE);
        pf_barrier(thread_pool);
/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.

# pragma omp for
      for ( i = 1; i < M - 1; i++ )
      {
        for ( j = 1; j < N - 1; j++ )
        {
          w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
        }
      }
    */
        parallel_for_cyclic(1, M - 1, 1, [](void* pf_args) -> void*{
            double (*_w)[N] = PF_GET_PARG(pf_args, w_u_arg_t) -> w;
            double (*_u)[N] = PF_GET_PARG(pf_args, w_u_arg_t) -> u;
            PF_FOR_LOOP(i, pf_args)
            {
                for (int j = 1; j < N - 1; j++ )
                {
                    _w[i][j] = ( _u[i-1][j] + _u[i+1][j] + _u[i][j-1] + _u[i][j+1] ) / 4.0;
                }
            }
            return NULL;
        }, (void*)&w_u_arg, thread_pool, CYCLIC_SIZE);
        pf_barrier(thread_pool);

/*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.

    diff = 0.0;
# pragma omp parallel shared ( diff, u, w ) private ( i, j, my_diff )
    {
      my_diff = 0.0;
# pragma omp for
      for ( i = 1; i < M - 1; i++ )
      {
        for ( j = 1; j < N - 1; j++ )
        {
          if ( my_diff < fabs ( w[i][j] - u[i][j] ) )
          {
            my_diff = fabs ( w[i][j] - u[i][j] );
          }
        }
      }
# pragma omp critical
      {
        if ( diff < my_diff )
        {
          diff = my_diff;
        }
      }
    }
    */
        diff = 0.0;
        typedef struct {
            double (*w)[N];
            double (*u)[N];
            double* p_diff;
        }w_u_diff_arg_t;
        w_u_diff_arg_t w_u_diff_arg = {w, u, &diff};
        parallel_for_cyclic(1, M - 1, 1, [](void* pf_args) -> void*{
            double (*_w)[N] = PF_GET_PARG(pf_args, w_u_diff_arg_t) -> w;
            double (*_u)[N] = PF_GET_PARG(pf_args, w_u_diff_arg_t) -> u;
            double* p_diff = PF_GET_PARG(pf_args, w_u_diff_arg_t) -> p_diff;
            double my_diff = *p_diff;

            PF_FOR_LOOP(i, pf_args)
            {
                for (int j = 1; j < N - 1; j++ )
                {
                    if ( my_diff < fabs ( _w[i][j] - _u[i][j] ) )
                    {
                        my_diff = fabs ( _w[i][j] - _u[i][j] );
                    }
                }
            }

            PF_CRITIAL_BEGIN(pf_args)
            if(*p_diff < my_diff)
            {
                *p_diff = my_diff;
            }
            PF_CRITIAL_END(pf_args)

            return NULL;
        }, (void*)&w_u_diff_arg, thread_pool, CYCLIC_SIZE);

        pf_barrier_with_master(thread_pool);

        iterations++;
        if ( iterations == iterations_print )
        {
          printf ( "  %8d  %f\n", iterations, diff );
          iterations_print = 2 *  iterations_print;
        }
    }
    wtime = get_wall_time() - wtime;

    printf ( "\n" );
    printf ( "  %8d  %f\n", iterations, diff );
    printf ( "\n" );
    printf ( "  Error tolerance achieved.\n" );
    printf ( "  Wallclock time = %f\n", wtime );
/*
  Terminate.
*/
    printf ( "\n" );
    printf ( "HEATED_PLATE_PARALLEL_FOR:\n" );
    printf ( "  Normal end of execution.\n" );

    pf_destroy_thread_pool(thread_pool);

    return 0;

# undef M
# undef N
}
