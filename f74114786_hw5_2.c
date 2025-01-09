#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#define MAXN 30
#define loc(a, b) ((a) * n + (b))

typedef struct
{
    int index;
    int x;
    int y;
} Point;

int compare(const void *a, const void *b)
{
    Point *pointA = (Point *)a;
    Point *pointB = (Point *)b;

    if (pointA->x == pointB->x)
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

void Andrew_monotone_chain(Point* points, Point* hull, int n, int* hull_size)
{
    qsort(points, n, sizeof(Point), compare);
    *hull_size = 0;
    for(int i = 0; i < n; ++i)
    {
        while(*hull_size >= 2 && cross(hull[(*hull_size) - 2], hull[(*hull_size) - 1], points[i]) >= 0)
        {
            (*hull_size)--;
        }
        hull[(*hull_size)++] = points[i];
    }
    for(int i = n - 2, t = *hull_size + 1; i >= 0; --i)
    {
        while(*hull_size >= t && cross(hull[(*hull_size) - 2], hull[(*hull_size) - 1], points[i]) >= 0)
        {
            (*hull_size)--;
        }
        hull[(*hull_size)++] = points[i];
    }
    *hull_size -= 1;
}

int main(int argc, char *argv[])
{
    int thread_count = atoi(argv[1]);
    char filename[100];
    scanf("%s", filename);
    FILE *file = fopen(filename, "r");
    int n;
    fscanf(file, "%d", &n);
    Point *points = malloc(n * sizeof(Point));
    for (int i = 0; i < n; ++i)
    {
        fscanf(file, "%d %d", &points[i].x, &points[i].y);
        points[i].index = i;
    }
    fclose(file);

    Point *hull = malloc(n * sizeof(Point));
    int hull_size = 0;
    int *distances = malloc(n * n * sizeof(int));

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                distances[loc(i, j)] = 0;
                continue;
            }
            distances[loc(i, j)] = sqrt(pow(points[i].x - points[j].x, 2) + pow(points[i].y - points[j].y, 2)) * 10000; // 4 decimal
        }
    }

    Andrew_monotone_chain(points, hull, n, &hull_size);

    uint32_t hull_bitmap = 0;
    for(int i = 0; i < hull_size; ++i)
    {
        hull_bitmap |= 1 << hull[i].index;
    }

    uint32_t global_min = UINT32_MAX;

    #pragma omp parallel for num_threads(thread_count) reduction(min:global_min)
    for(uint32_t i = 0; i < (1 << n); ++i)
    {
        if((i & hull_bitmap) != hull_bitmap)
        {
            continue;
        }
        uint32_t total_distance = 0;
        int keys[MAXN];
        int parent[MAXN];
        int mst[MAXN];
        for(int j = 0; j < n; ++j)
        {
            keys[j] = INT_MAX;
            mst[j] = 0;
            parent[j] = -1;
        }

        // start from hull[0]
        keys[hull[0].index] = 0;
        for(int j = 0; j < n; ++j)
        {
            int u = -1;

            // find minimum key
            for(int k = 0; k < n; ++k)
            {
                if(!mst[k] && (i & (1 << k)) && (u == -1 || keys[k] < keys[u]))
                {
                    u = k;
                }
            }
            if(u == -1)
            {
                continue;
            }
            mst[u] = 1;

            // update key
            for(int v = 0; v < n; ++v)
            {
                if(!mst[v] && (i & (1 << v)) && distances[loc(u, v)] < keys[v])
                {
                    keys[v] = distances[loc(u, v)];
                    parent[v] = u;
                    // printf("relax %d %d %d\n", u, v, keys[v]);
                }
            }
        }
        for (int v = 0; v < n; ++v)
        {
            if (parent[v] != -1)
            {
                total_distance += distances[loc(parent[v], v)];
                // printf("%d - %d: %d\n", parent[v], v, distances[loc(parent[v], v)]);
            }
        }
        if(total_distance < global_min)
        {
            global_min = total_distance;
        }
    }

    printf("%.4lf", global_min / 10000.0);
    
    free(points);
    free(hull);
    free(distances);
    return 0;
}