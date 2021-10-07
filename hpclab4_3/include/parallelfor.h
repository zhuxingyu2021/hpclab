#ifndef PARALLELFOR_H
#define PARALLELFOR_H

#include <pthread.h>

typedef struct {
    int start;
    int end;
    int increment;
    void* user_defined_arg = NULL;

    int num_threads;
    pthread_mutex_t* critial_mutex = NULL;

    //Initialized by worker in thread pool implementation
    int my_rank;
}parallel_for_arg_t;

typedef parallel_for_arg_t *p_parallel_for_arg_t;

typedef struct{
    parallel_for_arg_t parallel_for_arg;
    void* (*func)(void*);
}pf_thread_pool_arg_t;

typedef struct{
    int n_thread;
    pthread_t* threads;
    void* initialize_args;

    pf_thread_pool_arg_t* task_list;
    int task_list_count;
    int task_list_end;
    pthread_mutex_t task_list_mutex;
    pthread_cond_t task_list_empty_cond;
    pthread_cond_t task_list_full_cond;

    pthread_mutex_t critial_mutex;
}pf_thread_pool_t;

void parallel_for(int start, int end, int increment, void *(*functor)(void*),void *arg, int num_threads);

#define PF_GET_INDEX_START(p_x) (((p_parallel_for_arg_t)p_x)->start)
#define PF_GET_INDEX_END(p_x) (((p_parallel_for_arg_t)p_x)->end)
#define PF_GET_INDEX_INCREMENT(p_x) (((p_parallel_for_arg_t)p_x)->increment)
#define PF_GET_ARG(p_x, arg_t) ((arg_t)(((p_parallel_for_arg_t)p_x)->user_defined_arg))
#define PF_GET_PARG(p_x, u_arg_t) ((u_arg_t*)(((p_parallel_for_arg_t)p_x)->user_defined_arg))
#define PF_MYRANK(p_x) (((p_parallel_for_arg_t)p_x)->my_rank)
#define PF_NUM_THREADS(p_x) (((p_parallel_for_arg_t)p_x)->num_threads)

#define PF_CRITIAL_BEGIN(p_x) pthread_mutex_lock(((p_parallel_for_arg_t)p_x)->critial_mutex);
#define PF_CRITIAL_END(p_x) pthread_mutex_unlock(((p_parallel_for_arg_t)p_x)->critial_mutex);

#endif
