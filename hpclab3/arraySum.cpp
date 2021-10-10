#include <iostream>
#include "cmdline.h"
#include <ctime>
#include <cstdlib>
#include "mytime.h"
#include <pthread.h>

#define ARRAY_NUMMAX 10000
#define ARRAY_SIZE 1000

using namespace std;

int my_array[ARRAY_SIZE];
int global_index = 0;
pthread_mutex_t global_index_mutex;

typedef struct{
    int d;
    pthread_mutex_t m;
}mutex_int;

void* arraysum_solution1(void* sum);
void* arraysum_solution2(void* sum);

int main(int argc, char** argv)
{
    cmdline::parser cmdparser;
    cmdparser.add<int>("num_workers",'n',"the number of threads",true,1,cmdline::range(1,16));
    cmdparser.parse_check(argc, argv);

    int n_threads = cmdparser.get<int>("num_workers");
    double timestart,timeend;

    srand(time(0));
    for(int i = 0; i < ARRAY_SIZE; i++) my_array[i] = rand() % ARRAY_NUMMAX;

    pthread_mutex_init(&global_index_mutex, NULL);
    pthread_t threads[n_threads];

    timestart = get_wall_time();
    global_index = 0;
    mutex_int sum1;
    sum1.d = 0;
    pthread_mutex_init(&sum1.m, NULL);
    for(int i = 0; i < n_threads; i++) pthread_create(&threads[i], NULL, arraysum_solution1, (void*)&sum1);
    for(int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&sum1.m);
    timeend = get_wall_time();
    double solution1_time = timeend - timestart;

    timestart = get_wall_time();
    global_index = 0;
    mutex_int sum2;
    sum2.d = 0;
    pthread_mutex_init(&sum2.m, NULL);
    for(int i = 0; i < n_threads; i++) pthread_create(&threads[i], NULL, arraysum_solution2, (void*)&sum2);
    for(int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&sum2.m);
    timeend = get_wall_time();
    double solution2_time = timeend - timestart;

    cout << "Solution1:  Result:" << sum1.d << ", Time:" << solution1_time << "s" << endl;
    cout << "Solution2:  Result:" << sum2.d << ", Time:" << solution2_time << "s" << endl;

    pthread_mutex_destroy(&global_index_mutex);
    return 0;
}

void* arraysum_solution1(void* sum)
{
    mutex_int* global_sum = (mutex_int*) sum;
    register int mysum = 0;
    register int data;
    register int myindex;

    for( ; ; ){
        pthread_mutex_lock(&global_index_mutex);
        if(global_index < ARRAY_SIZE)
        {
            myindex = global_index++;
            pthread_mutex_unlock(&global_index_mutex);
        }
        else{
            pthread_mutex_unlock(&global_index_mutex);
            break;
        }

        data = my_array[myindex];
        mysum += data;
    }

    pthread_mutex_lock(&global_sum->m);
    global_sum->d += mysum;
    pthread_mutex_unlock(&global_sum->m);

    return NULL;
}

void* arraysum_solution2(void* sum)
{
    mutex_int* global_sum = (mutex_int*) sum;
    register int mysum = 0;
    register int myindex;

    for( ; ; ){
        pthread_mutex_lock(&global_index_mutex);
        if(global_index < ARRAY_SIZE)
        {
            myindex = global_index;
            global_index += 10;
            pthread_mutex_unlock(&global_index_mutex);
        }
        else{
            pthread_mutex_unlock(&global_index_mutex);
            break;
        }

        for(int i = 0; i < 10 && myindex < ARRAY_SIZE; i++, myindex++){
            mysum += my_array[myindex];
        }
    }

    pthread_mutex_lock(&global_sum->m);
    global_sum->d += mysum;
    pthread_mutex_unlock(&global_sum->m);

    return NULL;
}
