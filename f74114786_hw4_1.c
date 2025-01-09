#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define bound(x, n) ((x) < 0 ? (x) % (n) + (n) : (x) % (n))
#define addr(x, y, m) ((x) * (m) + (y))
#define next_process(rank, work_process) ((rank + 1) % work_process)
#define prev_process(rank, work_process) ((rank - 1 + work_process) % work_process)

void* next(void*);

long thread_count;
int work_threads;
int* matrix_a;
int* matrix_next_a;
int* matrix_k;
int n, m, dh, dw;
int iters;
int counter = 0;
pthread_mutex_t mutex;
pthread_cond_t cond_var;

int main(int argc, char* argv[])
{
    thread_count = strtol(argv[1], NULL, 10);
    pthread_t* thread_handles;

    // read input
    char filename[100];
    scanf("%s", filename);
    FILE* file = fopen(filename, "r");
    fscanf(file, "%d", &iters);
    fscanf(file, "%d %d", &n, &m);

    matrix_a = (int*)malloc(n * m * sizeof(int));
    matrix_next_a = (int*)malloc(n * m * sizeof(int));
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
        {
            fscanf(file, "%d", &matrix_a[addr(i, j, m)]);
        }
    }

    fscanf(file, "%d %d", &dh, &dw);
    matrix_k = (int*)malloc(dw * dh * sizeof(int));
    for(int i = 0; i < dh; ++i)
    {
        for(int j = 0; j < dw; ++j)
        {
            fscanf(file, "%d", &matrix_k[addr(i, j, dw)]);
        }
    }
    fclose(file);

    int padding_size = (dh - 1) / 2;    // padding size in row
    int max_threads_to_work = n / padding_size;
    work_threads = max_threads_to_work < thread_count ? max_threads_to_work : thread_count;

    // init mutex and cond var
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_var, NULL);

    // create threads
    thread_handles = (pthread_t*)malloc(work_threads * sizeof(pthread_t));
    int* thread_ranks = (int*)malloc(work_threads * sizeof(int));
    for(int i = 0; i < work_threads; ++i)
    {
        thread_ranks[i] = i;
        pthread_create(&thread_handles[i], NULL, next, (void*)&thread_ranks[i]);
    }

    for(int i = 0; i < work_threads; ++i)
    {
        pthread_join(thread_handles[i], NULL);
    }

    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
        {
            printf("%d ", matrix_a[addr(i, j, m)]);
        }
    }

    // destroy mutex and cond var
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_var);

    // free memory
    free(matrix_a);
    free(matrix_next_a);
    free(matrix_k);
    free(thread_handles);
    return 0;
}

void* next(void* my_rank)
{
    int rank = *(int*)my_rank;
    // a: current matrix
    // k: kernel
    // next_a: next matrix
    // n, m: matrix size
    // d: kernel size
    int padding_size_h = (dh - 1) / 2;
    int padding_size_w = (dw - 1) / 2;

    int my_start_row = rank * (n / work_threads) + (rank < n % work_threads ? rank : n % work_threads);
    int my_end_row = my_start_row + n / work_threads + (rank < n % work_threads ? 1 : 0);

    // printf("Thread %ld: %d %d\n", rank, my_start_row, my_end_row);

    // matrix:
    // padding: top size rows
    // local matrix: n rows
    // padding: bottom size rows
    for(int iter = 0; iter < iters; ++iter)
    {
        for(int i = my_start_row; i < my_end_row; ++i)    // i: row
        {
            for(int j = 0; j < m; ++j)
            {
                matrix_next_a[addr(i, j, m)] = 0;
                for(int x = -padding_size_h; x <= padding_size_h; ++x)
                {
                    for(int y = -padding_size_w; y <= padding_size_w; ++y)
                    {
                        // next_a[i][j] += k[size + x][size + y] * a[bound(i + x, n)][bound(j + y, m)];
                        matrix_next_a[addr(bound(i, n), bound(j, m), m)] += matrix_k[addr(padding_size_h + x, padding_size_w + y, dw)] * matrix_a[addr(bound(i + x, n), bound(j + y, m), m)];
                    }
                }
                matrix_next_a[addr(i, j, m)] /= dw * dh;
            }
        }
        pthread_mutex_lock(&mutex);
        counter += 1;
        if(counter == work_threads)
        {
            counter = 0;
            // swap matrix a and next_a
            int* temp = matrix_a;
            matrix_a = matrix_next_a;
            matrix_next_a = temp;
            pthread_cond_broadcast(&cond_var);
        }
        else
        {
            while(pthread_cond_wait(&cond_var, &mutex) != 0);
        }
        pthread_mutex_unlock(&mutex);
    }
}