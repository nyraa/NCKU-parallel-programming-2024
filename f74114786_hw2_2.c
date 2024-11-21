#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#define MAX_N 50000
#define INF INT_MAX
#define INIT_CAPACITY 10

typedef struct
{
    int distance;
    int vertex;
} Vertex;

typedef struct
{
    int src;
    int weight;
} Edge;

typedef struct
{
    Vertex* vertexes;
    int size;
} MinHeap;

void swap(Vertex* a, Vertex* b)
{
    Vertex temp = *a;
    *a = *b;
    *b = temp;
}

void minHeapify(MinHeap* minHeap, int idx)
{
    if(minHeap->size == 0)
    {
        return;
    }
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    if(left < minHeap->size && minHeap->vertexes[left].distance < minHeap->vertexes[smallest].distance)
    {
        smallest = left;
    }
    if(right < minHeap->size && minHeap->vertexes[right].distance < minHeap->vertexes[smallest].distance)
    {
        smallest = right;
    }
    if(smallest != idx)
    {
        swap(&minHeap->vertexes[smallest], &minHeap->vertexes[idx]);
        minHeapify(minHeap, smallest);
    }
}

void buildHeap(MinHeap* minHeap)
{
    for(int i = minHeap->size / 2 - 1; i >= 0; --i)
    {
        minHeapify(minHeap, i);
    }
}

Vertex getMin(MinHeap* minHeap)
{
    if(minHeap->size == 0)
    {
        Vertex vertex;
        vertex.vertex = INT_MAX;
        vertex.distance = INF;
        return vertex;
    }
    return minHeap->vertexes[0];
}

void removeMin(MinHeap* minHeap)
{
    if(minHeap->size == 0)
    {
        return;
    }
    minHeap->vertexes[0] = minHeap->vertexes[minHeap->size - 1];
    minHeap->size -= 1;
    minHeapify(minHeap, 0);
    return;
}

int findWeight(Edge* flatten_weight, int* weight_index, int* weight_offset, int src, int dest)
{
    for(int i = weight_offset[dest]; i < weight_offset[dest] + weight_index[dest]; ++i)
    {
        if(flatten_weight[i].src == src)
        {
            // printf("src: %d, dest: %d, weight: %d\n", src, dest, flatten_weight[i].weight);
            return flatten_weight[i].weight;
        }
    }
    return INF;
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    int comm_vertex_count;
    int local_vertex_start, local_vertex_end, local_vertex_count;

    Edge** r0_weight;              // weight[dest]: all edges from any vertex to dest
    int* r0_weight_capacity;
    int* comm_weight_index;

    // for distribute weight
    Edge* comm_flatten_weight;
    int edge_count;

    if(my_rank == 0)
    {
        // read file
        char filename[100];
        scanf("%s", filename);
        FILE* file = fopen(filename, "r");
        fscanf(file, "%d", &comm_vertex_count);

        r0_weight = (Edge**)malloc(comm_vertex_count * sizeof(Edge*));
        r0_weight_capacity = (int*)malloc(comm_vertex_count * sizeof(int));
        comm_weight_index = (int*)malloc(comm_vertex_count * sizeof(int));
        for(int i = 0; i < comm_vertex_count; ++i)
        {
            r0_weight[i] = (Edge*)malloc(INIT_CAPACITY * sizeof(Edge));
            r0_weight_capacity[i] = INIT_CAPACITY;
            comm_weight_index[i] = 0;
        }
        int src, dest, weight;
        edge_count = 0;
        while(fscanf(file, "%d %d %d", &src, &dest, &weight) != EOF)
        {
            if(comm_weight_index[dest] == r0_weight_capacity[dest])
            {
                r0_weight_capacity[dest] *= 2;
                r0_weight[dest] = (Edge*)realloc(r0_weight[dest], r0_weight_capacity[dest] * sizeof(Edge));
            }
            r0_weight[dest][comm_weight_index[dest]].src = src;
            r0_weight[dest][comm_weight_index[dest]].weight = weight;
            comm_weight_index[dest] += 1;
            edge_count += 1;
        }
        fclose(file);

        // flatten weight
        int edge_count_index = 0;
        comm_flatten_weight = (Edge*)malloc(edge_count * sizeof(Edge));
        for(int i = 0; i < comm_vertex_count; ++i)
        {
            for(int j = 0; j < comm_weight_index[i]; ++j)
            {
                comm_flatten_weight[edge_count_index] = r0_weight[i][j];
                edge_count_index += 1;
            }
            free(r0_weight[i]);
        }
        free(r0_weight);
        free(r0_weight_capacity);
    }

    // broadcast comm_n, edge_count
    MPI_Bcast(&comm_vertex_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(my_rank != 0)
    {
        comm_flatten_weight = (Edge*)malloc(edge_count * sizeof(Edge));
        comm_weight_index = (int*)malloc(comm_vertex_count * sizeof(int));
    }

    // broadcast comm_flatten_weight
    MPI_Bcast(comm_flatten_weight, edge_count, MPI_2INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(comm_weight_index, comm_vertex_count, MPI_INT, 0, MPI_COMM_WORLD);
    
    // calc local_vertex_start, local_vertex_end, local_vertex_count
    int quotient = comm_vertex_count / comm_sz;
    local_vertex_count = quotient + (my_rank < (comm_vertex_count % comm_sz));
    local_vertex_start = my_rank * quotient + (my_rank < (comm_vertex_count % comm_sz) ? my_rank : (comm_vertex_count % comm_sz));
    local_vertex_end = local_vertex_start + local_vertex_count;
    // printf("rank %d: local_vertex_start: %d, local_vertex_end: %d, local_vertex_count: %d\n", my_rank, local_vertex_start, local_vertex_end, local_vertex_count);

    // build weight offset
    int* weight_offset = (int*)malloc(comm_vertex_count * sizeof(int));
    weight_offset[0] = 0;
    for(int i = 1; i < comm_vertex_count; ++i)
    {
        weight_offset[i] = weight_offset[i - 1] + comm_weight_index[i - 1];
    }

    // dijkstra
    int* local_d = (int*)malloc(local_vertex_count * sizeof(int));
    MinHeap* minHeap = (MinHeap*)malloc(sizeof(MinHeap));
    minHeap->vertexes = (Vertex*)malloc(local_vertex_count * sizeof(Vertex));
    minHeap->size = local_vertex_count;
    for(int i = 0; i < local_vertex_count; ++i)
    {
        minHeap->vertexes[i].vertex = i + local_vertex_start;
        minHeap->vertexes[i].distance = i == 0 && my_rank == 0 ? 0 : INF;
        local_d[i] = i == 0 && my_rank == 0 ? 0 : INF;
    }

    while(1)
    {
        Vertex local_min_vertex = getMin(minHeap);
        Vertex global_min_vertex;
        MPI_Allreduce(&local_min_vertex, &global_min_vertex, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        // printf("rank %d: global_min_vertex: v%d w%d\n", my_rank, global_min_vertex.vertex, global_min_vertex.distance);
        if(global_min_vertex.vertex == INT_MAX) // all heap is empty
        {
            break;
        }

        if(global_min_vertex.vertex == local_min_vertex.vertex) // i am the global min node
        {
            removeMin(minHeap);
        }
        int global_min_vertex_index = global_min_vertex.vertex;
        int global_min_vertex_distance = global_min_vertex.distance;
        for(int i = 0; i < minHeap->size; ++i)
        {
            int local_vertex_index = minHeap->vertexes[i].vertex;   // start from 0
            int w = findWeight(comm_flatten_weight, comm_weight_index, weight_offset, global_min_vertex_index, local_vertex_index);
            if(w != INF && global_min_vertex_distance + w < local_d[local_vertex_index - local_vertex_start])
            {
                local_d[local_vertex_index - local_vertex_start] = global_min_vertex_distance + w;
                minHeap->vertexes[i].distance = global_min_vertex_distance + w;
            }
        }
        buildHeap(minHeap); // rebuild heap
    }

    // gather local_d
    int* global_d;
    if(my_rank == 0)
    {
        global_d = (int*)malloc(comm_vertex_count * sizeof(int));
    }
    int* sendcount = (int*)malloc(comm_sz * sizeof(int));
    int* displs = (int*)malloc(comm_sz * sizeof(int));
    for(int i = 0; i < comm_sz; ++i)
    {
        sendcount[i] = (quotient + (i < (comm_vertex_count % comm_sz)));
        displs[i] = i == 0 ? 0 : displs[i - 1] + sendcount[i - 1];
    }
    MPI_Gatherv(local_d, local_vertex_count, MPI_INT, global_d, sendcount, displs, MPI_INT, 0, MPI_COMM_WORLD);
    if(my_rank == 0)
    {
        for(int i = 0; i < comm_vertex_count; ++i)
        {
            printf("%d ", global_d[i]);
        }
    }

    // free memory
    if(my_rank == 0)
    {
        free(global_d);
    }
    free(comm_weight_index);
    free(comm_flatten_weight);
    free(weight_offset);
    free(local_d);
    free(minHeap->vertexes);
    free(minHeap);
    free(sendcount);
    free(displs);
    MPI_Finalize();
    return 0;
}