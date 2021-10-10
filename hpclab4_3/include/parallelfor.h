#ifndef PARALLELFOR_H
#define PARALLELFOR_H

#include <pthread.h>

typedef struct {
    int start;
    int end;
    int increment;
    void* user_defined_arg;

    int num_threads;
    pthread_mutex_t* critial_mutex;

    //Initialized by worker in thread pool implementation
    int my_rank;
}parallel_for_arg_t;

typedef parallel_for_arg_t *p_parallel_for_arg_t;


typedef struct{
    parallel_for_arg_t parallel_for_arg;
    int item_using;

    void* (*func)(void*);
    enum empty_task_type_t{
        PF_BARRIER_TASK, // 用于实现barrier屏障
        PF_KILL_TASK // 用于主线程结束worker线程
    }empty_task_type;
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
    pthread_cond_t task_list_not_empty_cond;

    int barrier_count;
    pthread_mutex_t barrier_mutex;
    pthread_cond_t barrier_cond;

    pthread_mutex_t critial_mutex;
}pf_thread_pool_t;

pf_thread_pool_t* pf_create_thread_pool(int num_threads);
void pf_destroy_thread_pool(pf_thread_pool_t* thread_pool);

#define parallel_for parallel_for_default

void parallel_for_default(int start, int end, int increment, void *(*functor)(void*),void *arg, int num_threads);
void parallel_for_default(int start, int end, int increment, void *(*functor)(void*),void *arg, pf_thread_pool_t* thread_pool);
void parallel_for_cyclic(int start, int end, int increment, void *(*functor)(void*),void *arg,
                         pf_thread_pool_t* thread_pool, int cyclic_sz);

void pf_wait_until_task_list_empty(pf_thread_pool_t* thread_pool);
void pf_barrier_with_master(pf_thread_pool_t* thread_pool);
void pf_barrier(pf_thread_pool_t* thread_pool);

#define PF_GET_INDEX_START(p_x) (((p_parallel_for_arg_t)p_x)->start)
#define PF_GET_INDEX_END(p_x) (((p_parallel_for_arg_t)p_x)->end)
#define PF_GET_INDEX_INCREMENT(p_x) (((p_parallel_for_arg_t)p_x)->increment)
#define PF_GET_ARG(p_x, arg_t) ((arg_t)(((p_parallel_for_arg_t)p_x)->user_defined_arg))
#define PF_GET_PARG(p_x, u_arg_t) ((u_arg_t*)(((p_parallel_for_arg_t)p_x)->user_defined_arg))
#define PF_MYRANK(p_x) (((p_parallel_for_arg_t)p_x)->my_rank)
#define PF_NUM_THREADS(p_x) (((p_parallel_for_arg_t)p_x)->num_threads)

#define PF_CRITIAL_BEGIN(p_x) pthread_mutex_lock(((p_parallel_for_arg_t)p_x)->critial_mutex);
#define PF_CRITIAL_END(p_x) pthread_mutex_unlock(((p_parallel_for_arg_t)p_x)->critial_mutex);

#define PF_FOR_LOOP(v_index, p_x) for (int v_index = PF_GET_INDEX_START(p_x); v_index < PF_GET_INDEX_END(p_x); v_index += PF_GET_INDEX_INCREMENT(p_x))

#endif
