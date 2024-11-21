#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#define next_index(current, size) ((current + 1) % size)
#define prev_index(current, size) ((current - 1 + size) % size)

typedef struct
{
    int x, y;
    int rank;
} Point;

int compare(const void *a, const void *b)
{
    Point *pointA = (Point *)a;
    Point *pointB = (Point *)b;

    if(pointA->x == pointB->x)
    {
        return pointA->y - pointB->y;
    }
    return pointA->x - pointB->x;
}

double cross(Point o, Point a, Point b)
{
    // OA to OB is clockwise -> positive
	return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}
Point local_points[12000];
int local_point_num;

Point result_hull[12000];
int result_hull_size;

Point local_hull[12000];
int local_hull_size;

Point received_hull[12000];
int received_hull_size;
int my_rank, comm_size;

void Andrew_monotone_chain()
{
    int hull_size = 0;
    for(int i = 0; i < local_point_num; ++i)
    {
        while(hull_size >= 2 && cross(local_hull[hull_size - 2], local_hull[hull_size - 1], local_points[i]) > 0)
        {
            hull_size--;
        }
        local_hull[hull_size++] = local_points[i];
    }
    for(int i = local_point_num - 2, t = hull_size + 1; i >= 0; --i)
    {
        while(hull_size >= t && cross(local_hull[hull_size - 2], local_hull[hull_size - 1], local_points[i]) > 0)
        {
            hull_size--;
        }
        local_hull[hull_size++] = local_points[i];
    }
    local_hull_size = hull_size - 1;
}

void merge_hull(Point* left, int left_size, Point* right, int right_size, Point* result_hull, int* result_size)
{
    // printf("rank %d: left_size = %d, right_size = %d\n", my_rank, left_size, right_size);
    if(left_size <= 0 && right_size > 0)
    {
        for(int i = 0; i < right_size; ++i)
        {
            result_hull[i] = right[i];
        }
        *result_size = right_size;
        return;
    }
    else if(left_size > 0 && right_size <= 0)
    {
        for(int i = 0; i < left_size; ++i)
        {
            result_hull[i] = left[i];
        }
        *result_size = left_size;
        return;
    }
    else if(left_size <= 0 && right_size <= 0)
    {
        *result_size = 0;
        return;
    }
    // merge to hull1
    // index of hull1 max and hull2 min
    int left_max_index = 0, right_max_index = 0;
    // find x_max and x_min
    int x_max = left[0].x, x_min = right[0].x;
    for(int k = 1; k < left_size; ++k)
    {
        if(left[k].x > x_max)
        {
            x_max = left[k].x;
            left_max_index = k;
        }
    }
    // printf("rank: %d: left max index = %d, %d, %d\n", my_rank, left_max_index, left[left_max_index].x, left[left_max_index].y);
    for(int k = 1; k < right_size; ++k)
    {
        if(right[k].x < x_min)
        {
            x_min = right[k].x;
            right_max_index = k;
        }
    }
    // printf("rank: %d: right max index = %d, %d, %d\n", my_rank, right_max_index, right[right_max_index].x, right[right_max_index].y);

    _Bool done_flag = 0;

    // upper tangent
    // upper i(left) upper j(right)
    int up_left = left_max_index, up_right = right_max_index;
    while(!done_flag)
    {
        done_flag = 1;
        while(cross(right[up_right], left[up_left], left[next_index(up_left, left_size)]) > 0) // rotate if next is clockwise
        {
            up_left = next_index(up_left, left_size);
            // printf("rank: %d: left rotate to index %d\n", my_rank, up_left);
            done_flag = 0;
        }
        while(cross(left[up_left], right[up_right], right[prev_index(up_right, right_size)]) < 0)   // rotate if prev is counter clockwise
        {
            up_right = prev_index(up_right, right_size);
            // printf("rank: %d: right rotate to index %d\n", my_rank, up_right);
            done_flag = 0;
        }
    }

    // lower tangent
    // lower i(left) lower j(right)
    int low_left = left_max_index, low_right = right_max_index;
    done_flag = 0;
    while(!done_flag)
    {
        done_flag = 1;
        while(cross(left[low_left], right[low_right], right[next_index(low_right, right_size)]) > 0)
        {
            low_right = next_index(low_right, right_size);
            // printf("rank: %d: right rotate to index %d\n", my_rank, low_right);
            done_flag = 0;
        }
        while(cross(right[low_right], left[low_left], left[prev_index(low_left, left_size)]) < 0)
        {
            low_left = prev_index(low_left, left_size);
            // printf("rank: %d: left rotate to index %d\n", my_rank, low_left);
            done_flag = 0;
        }
    }

    // printf("rank: %d: up_left = %d, low_left = %d, up_right = %d, low_right = %d\n", my_rank, left[up_left].rank, left[low_left].rank, right[up_right].rank, right[low_right].rank);

    // merge
    int i = 0;
    int j = 0;
    result_hull[i++] = left[j];
    while(j != low_left)
    {
        j = next_index(j, left_size);
        result_hull[i++] = left[j];
    }

    j = low_right;
    result_hull[i++] = right[j];
    while(j != up_right)
    {
        j = next_index(j, right_size);
        result_hull[i++] = right[j];
    }

    if(up_left != 0)
    {
        
        j = up_left;
        result_hull[i++] = left[j];
        while(next_index(j, left_size) != 0)
        {
            j = next_index(j, left_size);
            result_hull[i++] = left[j];
        }
    }
    *result_size = i;
    // printf("rank: %d: result_size = %d\n", my_rank, *result_size);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // define the MPI_Datatype
    MPI_Datatype MPI_POINT;

    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint displacements[3];
    displacements[0] = offsetof(Point, x);
    displacements[1] = offsetof(Point, y);
    displacements[2] = offsetof(Point, rank);
    MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    Point points[12000];
    int sendcounts[comm_size];
    int displs[comm_size];
    int n;
    if(my_rank == 0)
    {
        char filename[100];
        scanf("%s", filename);
        FILE *file = fopen(filename, "r");
        fscanf(file, "%d", &n);
        for(int i = 0; i < n; i++)
        {
            fscanf(file, "%d %d", &points[i].x, &points[i].y);
            points[i].rank = i + 1;
        }
        fclose(file);
        qsort(points, n, sizeof(Point), compare);
        // debug
        // local_point_num = n;
        // Andrew_monotone_chain();
        // for(int i = 0; i < local_hull_size; i++)
        // {
        //     printf("%d\n", local_hull[i].rank);
        // }
        int quotient = n / comm_size;
        int remainder = n % comm_size;
        for(int i = 0; i < comm_size; i++)
        {
            sendcounts[i] = i < remainder ? quotient + 1 : quotient;
            displs[i] = i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1];
        }
    }
    // broadcast n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // distribute points
    local_point_num = (my_rank < n % comm_size) ? (n / comm_size + 1) : (n / comm_size);
    MPI_Scatterv(points, sendcounts, displs, MPI_POINT, local_points, local_point_num, MPI_POINT, 0, MPI_COMM_WORLD);
    // printf("rank %d: received points", my_rank);
    // for(int i = 0; i < local_point_num; i++)
    // {
    //     printf("(i = %d, x = %d, y = %d, rank = %d) ", i, local_points[i].x, local_points[i].y, local_points[i].rank);
    // }

    // do compute
    Andrew_monotone_chain();

    // reduce data
    int phase = 0;

    // phase = ceil(log2(comm_size))
    int m = comm_size - 1;
    while(m > 0)
    {
        m >>= 1;
        phase++;
    }

    Point* my_hull_p = local_hull;
    Point* recv_hull_p = received_hull;
    Point* res_hull_p = result_hull;

    for(int k = 0; k < phase; ++k)
    {
        int partner;
        int distance = 1 << k;

        if((my_rank % (distance << 1)) == distance)
        {
            partner = my_rank - distance;
            MPI_Send(&local_hull_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(my_hull_p, local_hull_size, MPI_POINT, partner, 0, MPI_COMM_WORLD);
            // printf("rank %d send to %d at phase %d, size is %d\n", my_rank, partner, k, local_hull_size);
        }
        else if((my_rank % (distance << 1)) == 0)
        {
            partner = my_rank + distance;
            MPI_Recv(&received_hull_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_hull_p, received_hull_size, MPI_POINT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // merge
            // merge_hull(my_hull_p, *my_hull_size_p, recv_hull_p, *recv_hull_size_p, res_hull_p, res_hull_size_p);
            merge_hull(my_hull_p, local_hull_size, recv_hull_p, received_hull_size, res_hull_p, &result_hull_size);
            
            Point* tmp = my_hull_p;
            my_hull_p = res_hull_p;
            res_hull_p = tmp;
            local_hull_size = result_hull_size;   // debug, should be res_hull_size

            // printf("rank %d receive from %d at phase %d, size is %d\n", my_rank, partner, k, received_hull_size);
        }
    }

    if(my_rank == 0)
    {
        for(int i = 0; i < local_hull_size; i++)
        {
            printf("%d ", my_hull_p[i].rank);
        }
    }


    // release the MPI_Datatype
    MPI_Type_free(&MPI_POINT);

    MPI_Finalize();
    return 0;
}