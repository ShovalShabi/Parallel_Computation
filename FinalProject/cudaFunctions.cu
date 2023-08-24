#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * @brief Calculate the distance between two points.
 * @param p1 First point.
 * @param p2 Second point.
 * @param t T value.
 * @return Distance between the two points.
 */
__device__ double calcDistance(const Point p1, const Point p2, double t){
    double x1 = ((p1.x2 - p1.x1) / 2) * __sinf(t * M_PI / 2) + ((p1.x2 + p1.x1) / 2);
    double y1 = p1.a * x1 + p1.b;

    double x2 = ((p2.x2 - p2.x1) / 2) * __sinf(t * M_PI / 2) + (p2.x2 + p2.x1) / 2;
    double y2 = p2.a * x2 + p2.b;

    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}
/**
 * @brief This function insert an id of Proximity Criteria point that has at least K neighbours with distance less then D
 * @param startingIndex the current tvalue in the round
 * @param tidsAndPidsDevice array of results
 * @param pointId A point that satisfies the condition
 */
__device__ void addProxPoint(int startingIndex, int *tidsAndPidsDevice, int pointId){
    for (int i = 0; i < CONSTRAINT; i++){
        int index = startingIndex * CONSTRAINT + i;
        int currentVal = tidsAndPidsDevice[index];
        if (currentVal == -1){
            int expected = -1;
            int desired = pointId;

            if (atomicCAS(&tidsAndPidsDevice[index], expected, desired) == expected) /*Atomic compare ans swap*/
                return;
        }
    }
}

/**
 * @brief this function will calcluate for each tValue if there are Proximity Criteria points.
 * Each thread will get a specific point and will verify that is has proximity with the other points .
 * Before getting to calculations htere is nedd to check if the spot under the specific tValues is already taken if it does then the thread exits the function.
 * @note CONSTRAINT is the number of deisred Proxmity Criteria points under specific t value.
 * @param pointsArrDevice array of all N points
 * @param numPoints numbers of points
 * @param currentTvalue current tValue
 * @param distance max distance to check
 * @param tidsAndPidsDevice array of result
 * @param proximity need at least K points that fulfill the condition of Poximity Criteria
 * @param currentTindex the current t witihin the for loop
*/
__global__ void calculateProximity(Point *pointsArrDevice, int numPoints, double currentTvalue, double distance, int* tidsAndPidsDevice, double proximity, int currentTindex){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int counter = 0;

    if (tid < numPoints && atomicAdd(&tidsAndPidsDevice[currentTindex * CONSTRAINT + CONSTRAINT - 1], 0) == -1){

        for (int i = 0; i < numPoints; i++){

            //Checking if the last position under specific t index is already taken, -1 is free otherwise taken
            if (atomicAdd(&tidsAndPidsDevice[currentTindex * CONSTRAINT + CONSTRAINT - 1], 0) != -1)
                return;

            //The tested point passed the validations to ProximityCriteria
            if (pointsArrDevice[i].id != pointsArrDevice[tid].id && calcDistance(pointsArrDevice[tid], pointsArrDevice[i], currentTvalue) < distance){
                counter++;
                if (counter == proximity)
                    break;
            }
        }
        //Adding the Proximity Criteria the tidsAndPids array
        if (counter == proximity){
            int pointId = pointsArrDevice[tid].id;                      /*Reriving the Proximity Criteria id*/
            addProxPoint(currentTindex, tidsAndPidsDevice, pointId); /*Update the tidsAndPidsArray on proximity Criteria Point that have K neighbours*/
        }
    }
}


/**
 * @param buf Buffer for memory allocation
 * @param size Amount of data that need to copy(Bytes)
 */
void allocateMemoryOnDevice(void **buf, size_t size){
    cudaError_t err = cudaMalloc(buf, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Cannot to allocate memory on device. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @param dest Pointer to the destination buffer
 * @param src  Pointer to the source buffer
 * @param size Amount of data that need to copy(Bytes)
 * @param direction Which direction to copy the memory
 */
void copyMemory(void *dest, void *src, size_t size, cudaMemcpyKind direction){

    cudaError_t err = cudaMemcpy(dest, src, size, direction);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Perform the GPU computation for finding proximity criteria.
 * @param numPoints Number of points.
 * @param proximity Proximity value.
 * @param distance Distance value.
 * @param numT Number of t values.
 * @param actualTValues Array of actual t values.
 * @param pointArr Array of points.
 * @param tidsAndPids array of tids and pids.
 * @return 0 on success.
 */
int computeOnGPU(int numPoints, int proximtity, double distance, int numT, double* actualTValues, Point* pointsArr, int* tidsAndPids)
{
    cudaError_t err = cudaSuccess;

    int threadPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (numPoints + threadPerBlock - 1) / threadPerBlock;  //Measured by the number of points
    Point* pointsArrDevice = NULL;
    int* tidsAndPidsDevice = NULL;

    /*Allocating memory on gpu section*/
    allocateMemoryOnDevice((void **)&tidsAndPidsDevice, CONSTRAINT * numT * sizeof(int));
    allocateMemoryOnDevice((void **)&pointsArrDevice, numPoints * sizeof(Point));


    /*Copy memory to Device section*/
    copyMemory(pointsArrDevice, pointsArr, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    copyMemory(tidsAndPidsDevice, tidsAndPids, numT * CONSTRAINT * sizeof(int), cudaMemcpyHostToDevice);

    /*for each tvalue we will send it to GPU to compute the data and save it on proximites array*/
    for (int i = 0; i < numT; i++)
    {
        calculateProximity<<<numBlocks, threadPerBlock>>>(pointsArrDevice, numPoints, actualTValues[i], distance, tidsAndPidsDevice, proximtity, i);
        err = cudaGetLastError();
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to lanch calculateProximity kernel. -%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    //Copying the array that holds the Proximty Criteria to each matched t index
    copyMemory(tidsAndPids,tidsAndPidsDevice,numT * CONSTRAINT * sizeof(int), cudaMemcpyDeviceToHost);

    /*Freeing memoery back to device section*/
    if (cudaFree(pointsArrDevice) != cudaSuccess){
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(tidsAndPidsDevice) != cudaSuccess){
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
}