#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define bound(x, n) ((x) < 0 ? (x) % (n) + (n) : (x) % (n))
#define addr(x, y, m) ((x) * (m) + (y))
#define next_process(rank, work_process) ((rank + 1) % work_process)
#define prev_process(rank, work_process) ((rank - 1 + work_process) % work_process)

void next(int* a, int* next_a, int* k, int n, int m, int d)
{
    // a: current matrix
    // k: kernel
    // next_a: next matrix
    // n, m: matrix size
    // d: kernel size
    int padding_size = (d - 1) / 2;

    // matrix:
    // padding: top size rows
    // local matrix: n rows
    // padding: bottom size rows
    for(int i = 0; i < n; ++i)    // i: row
    {
        for(int j = 0; j < m; ++j)
        {
            next_a[addr(i, j, m)] = 0;
            for(int x = -padding_size; x <= padding_size; ++x)
            {
                for(int y = -padding_size; y <= padding_size; ++y)
                {
                    // next_a[i][j] += k[size + x][size + y] * a[bound(i + x, n)][bound(j + y, m)];
                    next_a[addr(bound(i, n), bound(j, m), m)] += k[addr(padding_size + x, padding_size + y, d)] * a[addr(bound(i + x, n), bound(j + y, m), m)];
                }
            }
            next_a[addr(i, j, m)] /= d * d;
        }
    }
}

int main(int argc, char* argv[])
{
    int my_rank, comm_sz;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int iters;
    int n, m, d;
    int* a;
    int* k;

    int* local_a;
    int* localnext_a;
    if(my_rank == 0)
    {
        char filename[100];
        scanf("%s", filename);
        FILE* file = fopen(filename, "r");
        fscanf(file, "%d", &iters);
        fscanf(file, "%d %d", &n, &m);

        a = (int*)malloc(n * m * sizeof(int));
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                fscanf(file, "%d", &a[addr(i, j, m)]);
            }
        }

        fscanf(file, "%d", &d);
        k = (int*)malloc(d * d * sizeof(int));
        for(int i = 0; i < d; ++i)
        {
            for(int j = 0; j < d; ++j)
            {
                fscanf(file, "%d", &k[addr(i, j, d)]);
            }
        }
        fclose(file);
    }

    // broadcast n, m, d, iters
    MPI_Bcast(&iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(my_rank != 0)
    {
        k = (int*)malloc(d * d * sizeof(int));
    }
    MPI_Bcast(k, d * d, MPI_INT, 0, MPI_COMM_WORLD);

    int padding_size = (d - 1) / 2; // padding size (in rows)
    int max_process = n / padding_size; // maximum number of processes that can have work
    int work_process = max_process < comm_sz ? max_process : comm_sz;   // number of processes that have work

    // create new communicator only containing work processes
    int color = my_rank < work_process ? 1 : MPI_UNDEFINED;
    MPI_Comm work_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, my_rank, &work_comm);

    // terminate if no work process
    if(my_rank >= work_process)
    {
        free(k);
        MPI_Finalize();
        return 0;
    }

    // scatter a
    int* sendcounts = (int*)malloc(work_process * sizeof(int));
    int* displs = (int*)malloc(work_process * sizeof(int));
    for(int i = 0; i < work_process; ++i)
    {
        sendcounts[i] = (n / work_process + (i < n % work_process ? 1 : 0)) * m;
        displs[i] = i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1];
    }
    int local_n = sendcounts[my_rank] / m;  // local n not including padding
    local_a = (int*)malloc(sendcounts[my_rank] * sizeof(int) + 2 * m * padding_size * sizeof(int));
    MPI_Scatterv(a, sendcounts, displs, MPI_INT, local_a + padding_size * m, sendcounts[my_rank], MPI_INT, 0, work_comm);
    localnext_a = (int*)malloc(sendcounts[my_rank] * sizeof(int) + 2 * m * padding_size * sizeof(int));


    for(int i = 0; i < iters; ++i)
    {
        // exchange padding
        if(work_process == 1)
        {
            memcpy(local_a, local_a + local_n * m, padding_size * m * sizeof(int));
            memcpy(local_a + (padding_size + local_n) * m, local_a + padding_size * m, padding_size * m * sizeof(int));
        }
        else
        {
            // send up padding to previous process
            MPI_Send(local_a + padding_size * m, padding_size * m, MPI_INT, prev_process(my_rank, work_process), 0, work_comm);
            // send down padding to next process
            MPI_Send(local_a + local_n * m, padding_size * m, MPI_INT, next_process(my_rank, work_process), 0, work_comm);
            // receive down padding from next process
            MPI_Recv(local_a + (padding_size + local_n) * m, padding_size * m, MPI_INT, next_process(my_rank, work_process), 0, work_comm, MPI_STATUS_IGNORE);
            // receive up padding from previous process
            MPI_Recv(local_a, padding_size * m, MPI_INT, prev_process(my_rank, work_process), 0, work_comm, MPI_STATUS_IGNORE);
        }
        next(local_a, localnext_a, k, local_n + 2 * padding_size, m, d);

        // swap local_a and localnext_a
        int* temp = local_a;
        local_a = localnext_a;
        localnext_a = temp;
    }

    // gather local_a to a
    MPI_Gatherv(local_a + padding_size * m, sendcounts[my_rank], MPI_INT, a, sendcounts, displs, MPI_INT, 0, work_comm);
    if(my_rank == 0)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                printf("%d ", a[addr(i, j, m)]);
            }
            // printf("\n");
        }
    }

    free(sendcounts);
    free(displs);
    free(local_a);
    free(localnext_a);
    free(k);
    if(my_rank == 0)
    {
        free(a);
    }

    MPI_Finalize();
    return 0;
}