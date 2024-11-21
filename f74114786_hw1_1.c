#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define SET_BIT(x, i) ((x) |= (1 << (i)))
#define CLEAR_BIT(x, i) ((x) &= ~(1 << (i)))

int n;      // program has n test parts
int m;      // program has m tests
uint32_t tests[32] = {0};   // 32-bit bit vector to represent the tests cover the program part
int costs[32] = {0};
int to;
int my_from;
int my_to;
int comm_sz, my_rank;

int do_test()
{
    int test_pass_count = 0;
    uint32_t ALL_PASS = (1 << n) - 1;
    if(to <= my_rank)
    {
        return 0;
    }
    int range_length = to / comm_sz;
    int range_remainder = to % comm_sz;

    my_from = my_rank * range_length + (my_rank < range_remainder ? my_rank : range_remainder);
    if(my_rank < range_remainder)
    {
        range_length += 1;
    }
    my_to = my_from + range_length - 1;
    // printf("my_rank: %d, my_from: %d, my_to: %d\n", my_rank, my_from, my_to);

    uint32_t test;
    uint32_t test_enable_vector = 0;
    for(int i = my_from; i <= my_to; ++i)
    {
        test_enable_vector = i;
        test = 0;
        for(int j = 0; j < m; ++j)
        {
            // check if the j-th bit is set
            if(test_enable_vector & 1)
            {
                test |= tests[j];
            }
            test_enable_vector >>= 1;
        }
        if(test == ALL_PASS)
        {
            // printf("hit test: %d\n", i);
            test_pass_count += 1;
        }
    }
    return test_pass_count;
}

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    

    if(my_rank == 0)
    {
        // main process
        char filename[100];
        scanf("%s", filename);
        FILE *fd = fopen(filename, "r");
        fscanf(fd, "%d %d", &n, &m);
        to = 1 << (m);    // tail not included, enum all tests
        int num_test_parts;
        int cost;       // cost of the test
        int test_num;   // test_num input for which test part
        int test;       // test is a bit vector
        for(int i = 0; i < m; ++i)
        {
            fscanf(fd, "%d %d", &num_test_parts, &cost);
            test = 0;
            for(int j = 0; j < num_test_parts; ++j)
            {
                fscanf(fd, "%d", &test_num);
                SET_BIT(test, test_num - 1);
            }
            tests[i] = test;
            // printf("test: %d\n", test);
        }
        fclose(fd);

        // send not include self
        for(int i = 1; i < comm_sz; ++i)
        {
            MPI_Send(&to, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(tests, 32, MPI_UINT32_T, i, 0, MPI_COMM_WORLD);
        }

        // master also do the test, don't be lazy
        int result = do_test();
        for(int i = 1; i < comm_sz; ++i)
        {
            int worker_result;
            MPI_Recv(&worker_result, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += worker_result;
        }
        printf("%d", result);
    }
    else
    {
        // worker process
        MPI_Recv(&to, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(tests, 32, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int result = do_test();
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}