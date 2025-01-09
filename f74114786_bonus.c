#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <limits.h>
#define MAX_CHAIN_LENGTH 10001     // = num_matrices + 1

unsigned long long dp[MAX_CHAIN_LENGTH][MAX_CHAIN_LENGTH];
int dimensions[MAX_CHAIN_LENGTH];
int num_matrices;
long thread_count;

// for barrier
int counter = 0;
pthread_mutex_t mutex;
pthread_cond_t cond;

void* matrix_chain_multiplication(void* arg)
{
    int my_rank = *((int*)arg);

    for(int len = 2; len <= num_matrices; ++len)
    {
        for(int start = my_rank + 1; start <= num_matrices - len + 1; start += thread_count)
        {
            int end = start + len - 1;
            dp[start][end] = INT_MAX;
            for(int split = start; split < end; ++split)
            {
                int cost = dp[start][split] + dp[split+1][end] + dimensions[start-1] * dimensions[split] * dimensions[end];
                if(cost < dp[start][end])
                {
                    dp[start][end] = cost;
                }
            }
        }

        // barrier
        pthread_mutex_lock(&mutex);
        counter += 1;
        if(counter == thread_count)
        {
            counter = 0;
            // printf("Chain length: %d\n", len);
            pthread_cond_broadcast(&cond);
        }
        else
        {
            while(pthread_cond_wait(&cond, &mutex) != 0);
        }
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(int argc, char* argv[])
{
    // read input
    char filename[100];
    scanf("%s", filename);
    FILE* file = fopen(filename, "r");
    fscanf(file, "%d", &num_matrices);
    for(int i = 0; i < num_matrices; i++)
    {
        fscanf(file, "%d", &dimensions[i]);
    }
    dimensions[num_matrices] = 1;
    fclose(file);

    // init dp table
    for(int i = 1; i <= num_matrices; i++)
    {
        dp[i][i] = 0;
    }

    // init mutex and cond
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);

    // create threads
    thread_count = strtol(argv[1], NULL, 10);
    
    pthread_t* threads = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
    int* thread_args = (int*)malloc(thread_count * sizeof(int));
    for(int i = 0; i < thread_count; ++i)
    {
        thread_args[i] = i;
        pthread_create(&threads[i], NULL, matrix_chain_multiplication, &thread_args[i]);
    }
    for(int i = 0; i < thread_count; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    printf("%lld", dp[1][num_matrices]);

    // destroy mutex and cond
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    // free memory
    free(threads);
    free(thread_args);
    return 0;
}