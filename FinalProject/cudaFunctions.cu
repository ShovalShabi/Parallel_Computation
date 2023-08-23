#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * @brief this function will calc the distance between two points
 * @param p1 Struct that points on one point.
 * @param p2 Struct that points on the secound point.
 * @param t The current tvalue
 */
__device__ double calcDistance(const Point p1, const Point p2, double t)
{
    double x1 = ((p1.x2 - p1.x1) / 2) * sin(t * M_PI / 2) + ((p1.x2 + p1.x1) / 2);
    double y1 = p1.a * x1 + p1.b;

    double x2 = ((p2.x2 - p2.x1) / 2) * sin(t * M_PI / 2) + (p2.x2 + p2.x1) / 2;
    double y2 = p2.a * x2 + p2.b;

    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}
/**
 * @brief In this function points that have at least K points that the distance less then D will enter to the function
 * and update the array of results.
 * @param startingIndex the current tvalue in the round
 * @param proximites array of results
 * @param pointId A point that satisfies the condition
 */
__device__ void updateProximitePoints(int startingIndex, int *proximites, int pointId)
{
    for (int i = 0; i < CONSTRAINT; i++)
    {
        int index = startingIndex * CONSTRAINT + i;
        int currentVal = proximites[index];
        if (currentVal == -1)
        {
            int expected = -1;
            int desired = pointId;

            if (atomicCAS(&proximites[index], expected, desired) == expected) /*Fill the proximites[index] in atomic way*/
            {
                return;
            }
        }
    }
}
/**
 * @brief this function will calcluate for each Tvalue[tIndex] Checks if it exists a Proximity Criteria.
 * Each thread will get point and check with other points if there is another point if the distance is smaller then D.
 * but before checking it, we will check the last poistion of d_proximities[tIndex * CONSTRAINT + CONSTRAINT - 1].
 * explanation: All CONSTRAINT cells represent K points that Proximity Criteria. if there at Least K points fulfill the condition the points will
 * enter to the d_proximities.
 * Note: if one of the threads fill the last point in  d_proximities[tIndex * CONSTRAINT + CONSTRAINT - 1] the other threads will not continue
 * calclute the distance.
 * @param d_points array of all N points
 * @param N numbers of points
 * @param tValue current tValue
 * @param D max distance to check
 * @param d_proximites array of result
 * @param K need at least K points that fulfill the condition of Poximity Criteria
 * @param tIndex the current t in the for loop


*/
__global__ void calculateProximity(Point *d_points, int N, double tValue, double D, int *d_proximities, double K, int tIndex)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int counter = 0;
    if (tid < N && atomicAdd(&d_proximities[tIndex * CONSTRAINT + CONSTRAINT - 1], 0) == -1)
    {
        for (int i = 0; i < N; i++)
        {

            if (atomicAdd(&d_proximities[tIndex * CONSTRAINT + CONSTRAINT - 1], 0) != -1)
            {
                return;
            }

            if (d_points[i].id != d_points[tid].id && calcDistance(d_points[tid], d_points[i], tValue) < D)
            {
                counter++;
                if (counter == K)
                {

                    int pointId = d_points[tid].id;                        /*This point is proximity*/
                    updateProximitePoints(tIndex, d_proximities, pointId); /*0,proximites,point that have 3 points*/
                    break;
                }
            }
        }
    }
}

/**
 * @brief Initialize the tids and pids array on the GPU, -1 means that the slot is free to assignment.
 * @param tidsAndPidsDevice Device array of tids and pids.
 */
__global__ void intializeTidsAndPids(int* tidsAndPidsDevice){
    int threadId = blockIdx.x;

    tidsAndPidsDevice[threadId] = -1;
}

void allocateMemDevice(void **ptr, size_t size)
{
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(ptr, size); /*Allocate mem on device*/
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cannot to allocate memory on device -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
/**
 * @param dest Pointer to the dest in device / host memory
 * @param src  Pointer to the src in host / device memory
 * @param size How much data need to copy(Bytes)
 * @param direction Which direction to copy the mem(host-> device) | (device -> host)
 */
void copyMemory(void *dest, void *src, size_t size, cudaMemcpyKind direction)
{
    /*Copy mem from device to host OR host to device depending on the direction*/
    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(dest, src, size, direction);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int computeOnGPU(int N, int K, double D, int chunkSize, double *tValues, Point *allPoints, int *proximities)
{
    cudaError_t err = cudaSuccess;

    int threadPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + threadPerBlock - 1) / threadPerBlock;
    Point *d_points = NULL;
    int *d_proximities = NULL;
    /*Allocating mem on gpu section*/
    allocateMemDevice((void **)&d_points, N * sizeof(Point));
    printf("here1\n");

    allocateMemDevice((void **)&d_proximities, CONSTRAINT * chunkSize * sizeof(int)); 
    printf("here2\n");

    /*End allocate mem*/

    /*Copy mem to Device section*/
    copyMemory(d_points, allPoints, N * sizeof(Point), cudaMemcpyHostToDevice);
    intializeTidsAndPids<<<CONSTRAINT*chunkSize,1>>>(proximities);
    // copyMemory(d_proximities, proximities, chunkSize * CONSTRAINT * sizeof(int), cudaMemcpyHostToDevice);
    /*End copy mem to Device*/

    /*for each tvalue we will send it to GPU to compute the data and save it on proximites array*/
    for (int i = 0; i < chunkSize; i++)
    {
        calculateProximity<<<blocksPerGrid, threadPerBlock>>>(d_points, N, tValues[i], D, d_proximities, K, i);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to lanch calculateProximity kernel. -%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    printf("here\n");

    // cudaDeviceSynchronize();


    // for (int i = 0; i < chunkSize; i++)
    // {
    //     for (int j = 0; j < CONSTRAINT; j++)
    //     {
    //         printf("device[%d][%d] = %d, actual[%d][%d] = %d\n",i,j,d_proximities[i*CONSTRAINT+j],i,j,proximities[i*CONSTRAINT+j]);
    //     }
        
    // }
    

    copyMemory(proximities, d_proximities, chunkSize * CONSTRAINT * sizeof(int), cudaMemcpyDeviceToHost);

    // err = cudaMemcpy(proximities, d_proximities, chunkSize * CONSTRAINT * sizeof(int), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    if (cudaFree(d_points) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_proximities) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
}