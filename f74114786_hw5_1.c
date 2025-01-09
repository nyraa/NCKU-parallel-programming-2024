#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#define MAX_TASKS 11
#define MAX_PERMUTATION 39916800
#define MAX_GANTT_CHART_LENGTH 300

int all_permutations(char src[MAX_TASKS], char dest[MAX_PERMUTATION][MAX_TASKS], int l, int r, int *index, int task_num)
{
    int size = 0;
    if(l == r)
    {
        for(int i = 0; i < task_num; ++i)
        {
            dest[*index][i] = src[i];
        }
        (*index)++;
        return 1;
    }
    else
    {
        short temp;
        for(int i = l; i <= r; ++i)
        {
            temp = src[l];
            src[l] = src[i];
            src[i] = temp;

            size += all_permutations(src, dest, l + 1, r, index, task_num);

            temp = src[l];
            src[l] = src[i];
            src[i] = temp;
        }
        return size;
    }
}

int main(int argc, char *argv[])
{
    int num_threads = atoi(argv[1]);
    char filename[100];
    scanf("%s", filename);
    FILE *file = fopen(filename, "r");


    char task_index[MAX_TASKS];
    char (*task_permutation)[11] = malloc(MAX_PERMUTATION * MAX_TASKS * sizeof(char));
    int index = 0;


    int task_num;
    fscanf(file, "%d", &task_num);

    int (*task)[3] = malloc(task_num * 3 * sizeof(int));
    int p, r, w;
    for(int i = 0; i < task_num; ++i)
    {
        task_index[i] = i;
        fscanf(file, "%d %d %d", &p, &r, &w);
        task[i][0] = p;
        task[i][1] = r;
        task[i][2] = w;
    }
    int permutation_size = all_permutations(task_index, task_permutation, 0, task_num - 1, &index, task_num);

    int global_min = INT_MAX;
    #pragma omp parallel
    {
        int local_min = INT_MAX;
        int local_cost;
        #pragma omp for
        for(int i = 0; i < permutation_size; ++i)
        {
            local_cost = 0;
            int gantt_chart[MAX_GANTT_CHART_LENGTH] = {0};
            for(int j = 0; j < task_num; ++j)
            {
                int curr_task_id = task_permutation[i][j];
                int curr_task_p = task[curr_task_id][0];
                int curr_task_r = task[curr_task_id][1];
                int curr_task_w = task[curr_task_id][2];

                while(curr_task_p)
                {
                    if(gantt_chart[curr_task_r] == 0)
                    {
                        gantt_chart[curr_task_r] = curr_task_id + 1;    // 0 in gantt present idle time
                        curr_task_p--;
                    }
                    curr_task_r++;
                }
                local_cost += curr_task_r * curr_task_w;
            }
            if(local_cost < local_min)
            {
                local_min = local_cost;
            }
        }
        #pragma omp critical
        {
            if(local_min < global_min)
            {
                global_min = local_min;
            }
        }
    }

    printf("%d", global_min);
        
    free(task);
    free(task_permutation);
    fclose(file);
    return 0;
}