#include <iostream>
#include <pthread.h>
#include <cmath>

using namespace std;

typedef struct{
    double a;
    double b;
    double c;
    double result[4];
}quadratic_equation_args;

struct{
    pthread_mutex_t barrier_mutex;
    pthread_cond_t barrier_cond;
    int count = 0;
}barrier_v;

#define ALGORITHM_NEED_THREADS 4

void* solve_minus_b(void* args);
void* solve_2a(void* args);
void* solve_b2(void* args);
void* solve_4ac(void* args);


int main()
{
    quadratic_equation_args argqe;
    cout << "Input a, b, c: " << endl;
    cin >> argqe.a >> argqe.b >> argqe.c;

    pthread_mutex_init(&barrier_v.barrier_mutex, NULL);
    pthread_cond_init(&barrier_v.barrier_cond, NULL);

    pthread_t thread0, thread1, thread2, thread3;
    pthread_create(&thread0, NULL, solve_minus_b, (void*)&argqe);
    pthread_create(&thread1, NULL, solve_2a, (void*)&argqe);
    pthread_create(&thread2, NULL, solve_b2, (void*)&argqe);
    pthread_create(&thread3, NULL, solve_4ac, (void*)&argqe);

    pthread_join(thread3, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread0, NULL);

    pthread_mutex_destroy(&barrier_v.barrier_mutex);
    pthread_cond_destroy(&barrier_v.barrier_cond);

    cout << "x1 = " << argqe.result[0] << ", x2 = " << argqe.result[1] << endl;

    return 0;
}

void* solve_minus_b(void* args)
{
    quadratic_equation_args* p_args = (quadratic_equation_args*)args;
    p_args->result[0] = -p_args->b;

    pthread_mutex_lock(&barrier_v.barrier_mutex);
    barrier_v.count++;
    if(barrier_v.count == ALGORITHM_NEED_THREADS) {
        pthread_cond_broadcast(&barrier_v.barrier_cond);
        barrier_v.count = 0;
    }else{
        while(pthread_cond_wait(&barrier_v.barrier_cond, &barrier_v.barrier_mutex) != 0);
    }
    pthread_mutex_unlock(&barrier_v.barrier_mutex);

    double in_sqrt = sqrt(p_args->result[2] - p_args->result[3]);
    double x0 = (p_args->result[0] + in_sqrt)/p_args->result[1];
    double x1 = (p_args->result[0] - in_sqrt)/p_args->result[1];
    p_args->result[0] = x0;
    p_args->result[1] = x1;
    return NULL;
}

void* solve_2a(void* args)
{
    quadratic_equation_args* p_args = (quadratic_equation_args*)args;
    p_args->result[1] = 2 * p_args->a;

    pthread_mutex_lock(&barrier_v.barrier_mutex);
    barrier_v.count++;
    if(barrier_v.count == ALGORITHM_NEED_THREADS) {
        pthread_cond_broadcast(&barrier_v.barrier_cond);
        barrier_v.count = 0;
    }else{
        while(pthread_cond_wait(&barrier_v.barrier_cond, &barrier_v.barrier_mutex) != 0);
    }
    pthread_mutex_unlock(&barrier_v.barrier_mutex);

    return NULL;
}

void* solve_b2(void* args)
{
    quadratic_equation_args* p_args = (quadratic_equation_args*)args;
    p_args->result[2] = p_args->b * p_args->b;

    pthread_mutex_lock(&barrier_v.barrier_mutex);
    barrier_v.count++;
    if(barrier_v.count == ALGORITHM_NEED_THREADS) {
        pthread_cond_broadcast(&barrier_v.barrier_cond);
        barrier_v.count = 0;
    }else{
        while(pthread_cond_wait(&barrier_v.barrier_cond, &barrier_v.barrier_mutex) != 0);
    }
    pthread_mutex_unlock(&barrier_v.barrier_mutex);

    return NULL;
}

void* solve_4ac(void* args)
{
    quadratic_equation_args* p_args = (quadratic_equation_args*)args;
    p_args->result[3] = 4 * p_args->a * p_args->c;

    pthread_mutex_lock(&barrier_v.barrier_mutex);
    barrier_v.count++;
    if(barrier_v.count == ALGORITHM_NEED_THREADS) {
        pthread_cond_broadcast(&barrier_v.barrier_cond);
        barrier_v.count = 0;
    }else{
        while(pthread_cond_wait(&barrier_v.barrier_cond, &barrier_v.barrier_mutex) != 0);
    }
    pthread_mutex_unlock(&barrier_v.barrier_mutex);

    return NULL;
}
