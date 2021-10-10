#include <iostream>
#include <pthread.h>
#include <time.h>
#include <iomanip>
#include "cmdline.h"
#include <random>
#include <sys/time.h>
#include "mytime.h"

#define N_ITER 1000

using namespace std;

typedef struct{
    long seed;
    long long local_n_iter;
    long long counts = 0;
}montecarlo_args;

void* montecarlo_workers(void*);

int main(int argc, char** argv)
{
    cmdline::parser cmdparser;
    cmdparser.add<int>("num_workers",'n',"the number of threads",true,1,cmdline::range(1,16));
    cmdparser.parse_check(argc, argv);
    int n_threads = cmdparser.get<int>("num_workers");

    pthread_t threads[n_threads];
    montecarlo_args args[n_threads];

    double timestart, timeend;

    timestart = get_wall_time();
    for(int i = 0; i < n_threads; i++){
        struct timeval timenow;
        gettimeofday(&timenow,NULL);

        if(i != n_threads - 1) {
            args[i].local_n_iter = N_ITER / n_threads;
        }
        else{
            args[i].local_n_iter = N_ITER - (N_ITER / n_threads) * (n_threads - 1);
        }
        args[i].seed = (long)timenow.tv_usec * 16 + i;
        //cout << args[i].seed << endl;

        pthread_create(&threads[i], NULL, montecarlo_workers, &args[i]);
    }
    long long global_counts = 0;
    for(int i = 0; i < n_threads; i++){
        pthread_join(threads[i], NULL);
        global_counts += args[i].counts;
        //cout << args[i].counts << endl;
    }
    timeend = get_wall_time();

    std::cout << std::fixed;
    cout << "Result by parallel monte carlo: " << setprecision(10) << double(global_counts) / double(N_ITER) << endl;
    cout << "Time cost by parallel monte carlo: " << timeend - timestart << "s" << endl;

    timestart = get_wall_time();

    default_random_engine random_e(time(0));
    uniform_real_distribution<double> uniform_rand(0, 1);
    double x, y;
    long long counts = 0;
    for(long long i = 0; i < N_ITER; i++){
        x = uniform_rand(random_e);
        y = uniform_rand(random_e);

        if(y < x * x) counts++;
    }
    timeend = get_wall_time();

    std::cout << std::fixed;
    cout << "Result by single threaad monte carlo:" << setprecision(10) << double(counts) / double(N_ITER) << endl;
    cout << "Time cost by single thread monte carlo:" << timeend - timestart << "s" << endl;

    return 0;
}

void* montecarlo_workers(void* args)
{
    long long* return_value = &((montecarlo_args*)args)->counts;
    long long local_n_iter = ((montecarlo_args*)args)->local_n_iter;
    long long counts_local = 0;

    default_random_engine random_e(((montecarlo_args*)args)->seed);
    uniform_real_distribution<double> uniform_rand(0, 1);

    double x, y;
    for(long long i = 0; i < local_n_iter; i++)
    {
        x = uniform_rand(random_e);
        y = uniform_rand(random_e);

        if(y < x * x) counts_local++;
    }

    *return_value = counts_local;
    return NULL;
}
