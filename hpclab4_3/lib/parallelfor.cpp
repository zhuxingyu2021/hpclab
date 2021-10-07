#include "parallelfor.h"
#include <stdlib.h>

#define TASK_LIST_MAX 100

void* thread_pool_worker(void* _pf_thread_pool);

void parallel_for(int start, int end, int increment, void *(*functor)(void*),void *arg, int num_threads)
{
    if(end <= start || increment <= 0) return;
    int n_blocks = (end - start -1)/increment + 1;

    pthread_t threads[num_threads];
    parallel_for_arg_t pf_args[num_threads];

    pthread_mutex_t critial_mutex;
    pthread_mutex_init(&critial_mutex,NULL);

    for(int i = 0; i < num_threads; i++){
            int n_comm_blocks = n_blocks/num_threads;
            pf_args[i].start = start + i * n_comm_blocks * increment;
            pf_args[i].increment = increment;
            pf_args[i].user_defined_arg = arg;

            if(i != num_threads - 1) pf_args[i].end = pf_args[i].start + n_comm_blocks * increment;
            else pf_args[i].end = end;

            pf_args[i].my_rank = i;
            pf_args[i].num_threads = num_threads;
            pf_args[i].critial_mutex = &critial_mutex;

            pthread_create(&threads[i], NULL, functor, &pf_args[i]);
    }

    for(int i = 0; i < num_threads; i++){
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&critial_mutex);
}

typedef struct{
    pf_thread_pool_t *thread_pool;
    int my_rank;
}pf_worker_arg_t;

pf_thread_pool_t* pf_create_thread_pool(int num_threads)
{
    pf_thread_pool_t *thread_pool = (pf_thread_pool_t *) malloc(sizeof(pf_thread_pool_t));
    thread_pool->n_thread = num_threads;
    thread_pool->threads = (pthread_t*) malloc(sizeof(pthread_t) * num_threads);
    thread_pool->task_list = (pf_thread_pool_arg_t*) malloc(sizeof(pf_thread_pool_arg_t) * TASK_LIST_MAX);
    thread_pool->task_list_count = 0;
    thread_pool->task_list_end = 0;
    thread_pool->initialize_args = malloc(sizeof(pf_worker_arg_t) * num_threads);
    pf_worker_arg_t* pf_worker_arg_list = (pf_worker_arg_t*)thread_pool->initialize_args;

    pthread_mutex_init(&thread_pool->task_list_mutex, NULL);
    pthread_mutex_init(&thread_pool->critial_mutex,NULL);
    pthread_cond_init(&thread_pool->task_list_empty_cond, NULL);
    pthread_cond_init(&thread_pool->task_list_full_cond, NULL);

    for(int i = 0; i < num_threads; i++)
    {
        pf_worker_arg_list[i].thread_pool = thread_pool;
        pf_worker_arg_list[i].my_rank = i;
        pthread_create(&thread_pool->threads[i], NULL, thread_pool_worker, (void*)&pf_worker_arg_list[i]);
    }

    return thread_pool;
}

void pf_destroy_thread_pool(pf_thread_pool_t* thread_pool)
{
    for(int i = 0; i < thread_pool->n_thread; i++)
    {
        pthread_mutex_lock(&thread_pool->task_list_mutex);
        if((thread_pool->task_list_end + 1) % TASK_LIST_MAX == thread_pool->task_list_count) //任务队列满
        {
            pthread_cond_wait(&thread_pool->task_list_full_cond, &thread_pool->task_list_mutex);
        }
        else{
            pf_thread_pool_arg_t &taskarg = thread_pool->task_list[thread_pool->task_list_end];
            taskarg.func = NULL;
            thread_pool->task_list_end = (thread_pool->task_list_end + 1) % TASK_LIST_MAX;
            pthread_cond_signal(&thread_pool->task_list_empty_cond);
        }
        pthread_mutex_unlock(&thread_pool->task_list_mutex);
    }
    for(int i = 0; i < thread_pool->n_thread; i++)
    {
        pthread_join(thread_pool->threads[i], NULL);
    }

    free(thread_pool->threads);
    free(thread_pool->task_list);
    free(thread_pool->initialize_args);
    pthread_mutex_destroy(&thread_pool->task_list_mutex);
    pthread_mutex_destroy(&thread_pool->critial_mutex);
    pthread_cond_destroy(&thread_pool->task_list_empty_cond);
    pthread_cond_destroy(&thread_pool->task_list_full_cond);

    free(thread_pool);
}

void parallel_for(int start, int end, int increment, void *(*functor)(void*),void *arg, pf_thread_pool_t* thread_pool)
{
    if(end <= start || increment <= 0) return;
    int n_blocks = (end - start -1)/increment + 1;
    int num_threads = thread_pool->n_thread;

    for(int i = 0; i < num_threads; i++){
        int n_comm_blocks = n_blocks/num_threads;

        pthread_mutex_lock(&thread_pool->task_list_mutex);
        if((thread_pool->task_list_end + 1) % TASK_LIST_MAX == thread_pool->task_list_count) //任务队列满
        {
            pthread_cond_wait(&thread_pool->task_list_full_cond, &thread_pool->task_list_mutex);
        }
        else{
            pf_thread_pool_arg_t &taskarg = thread_pool->task_list[thread_pool->task_list_end];
            taskarg.parallel_for_arg.start = start + i * n_comm_blocks * increment;
            taskarg.parallel_for_arg.increment = increment;
            taskarg.parallel_for_arg.user_defined_arg = arg;

            if (i != num_threads - 1)
                taskarg.parallel_for_arg.end = taskarg.parallel_for_arg.start + n_comm_blocks * increment;
            else taskarg.parallel_for_arg.end = end;

            taskarg.parallel_for_arg.num_threads = num_threads;
            taskarg.parallel_for_arg.critial_mutex = &thread_pool->critial_mutex;

            taskarg.func = functor;

            thread_pool->task_list_end = (thread_pool->task_list_end + 1) % TASK_LIST_MAX;
            pthread_cond_signal(&thread_pool->task_list_empty_cond);
        }

        pthread_mutex_unlock(&thread_pool->task_list_mutex);
    }
}

void* thread_pool_worker(void* _pf_worker_arg)
{
    pf_thread_pool_t* pf_thread_pool = ((pf_worker_arg_t *)_pf_worker_arg)->thread_pool;
    int my_rank = ((pf_worker_arg_t *)_pf_worker_arg)->my_rank;

    while(true)
    {
        pf_thread_pool_arg_t taskarg;
        pthread_mutex_lock(&pf_thread_pool->task_list_mutex);
        if(pf_thread_pool->task_list_count == pf_thread_pool->task_list_end) //任务队列为空
        {
            pthread_cond_wait(&pf_thread_pool->task_list_empty_cond, &pf_thread_pool->task_list_mutex);
        }
        else
        {
            if(pf_thread_pool->task_list[pf_thread_pool->task_list_count].func) {
                taskarg = pf_thread_pool->task_list[pf_thread_pool->task_list_count];
                pf_thread_pool->task_list_count = (pf_thread_pool->task_list_count + 1) % TASK_LIST_MAX;
                pthread_cond_signal(&pf_thread_pool->task_list_full_cond);
            }
            else{
                pf_thread_pool->task_list_count = (pf_thread_pool->task_list_count + 1) % TASK_LIST_MAX;
                pthread_cond_signal(&pf_thread_pool->task_list_full_cond);
                pthread_mutex_unlock(&pf_thread_pool->task_list_mutex);
                break;
            }
        }
        pthread_mutex_unlock(&pf_thread_pool->task_list_mutex);

        taskarg.parallel_for_arg.my_rank = my_rank;
        taskarg.func((void*)&taskarg.parallel_for_arg);
    }

    return NULL;
}
