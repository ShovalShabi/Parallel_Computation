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
__device__ double calculateDistance(const Point p1,const Point p2, double t){
    double xP1 = ((p1.x2 - p1.x1) / 2) * __sinf(t * M_PI / 2) + (p1.x2 + p1.x1) / 2;
    double yP1 = p1.a * xP1 + p1.b;

    double xP2 = ((p2.x2 - p2.x1) / 2) * __sinf(t * M_PI / 2) + (p2.x2 + p2.x1) / 2;
    double yP2 = p2.a * xP2 + p2.b;

    double dx = xP2 - xP1;
    double dy = yP2 - yP1;

    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Find the proximity criteria for each point and t value on the GPU.
 * @param pointsArrDevice Device array of points.
 * @param nCount Total number of points.
 * @param actualTsDevice Device array of actual t values.
 * @param tidAndPidsDevice Device array of tids and pids.
 * @param numT Total number of t values.
 * @param proximity Proximity value.
 * @param distance Distance value.
 * @param minTIndex Minimum t index.
 * @param maxTIndex Maximum t index.
 */
__global__ void findProximityCriteria(Point* pointsArrDevice, int nCount, double* actualTsDevice, int* tidAndPidsDevice, int numT, int proximity, double distance, int minTIndex, int maxTIndex) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < nCount * numT) {
        int indexPoint = threadId % nCount;  // The index of the point within the buffer
        int indexT = threadId / nCount;      // The index of the current t value

        if (tidAndPidsDevice[indexT* CONSTRAINT + CONSTRAINT -1] != -1)
            return;

        // Making sure that the process calculates only the range of t values that are assigned to it
        if (minTIndex + indexT <= maxTIndex) {

            Point currentPoint = pointsArrDevice[indexPoint];
            double currentT = actualTsDevice[indexT];

            int count = 0;

            for (int i = 0; i < nCount; i++) {
                if (i != indexPoint) {
                    Point otherPoint = pointsArrDevice[i];

                    double dist = calculateDistance(currentPoint, otherPoint, currentT);

                    if (dist <= distance)
                        count++;

                    if (count == proximity)
                        break;
                }
            }

            // The examined point reached K neighbors which is proximity values
            if (count == proximity) {
                int startIndex = indexT * CONSTRAINT;
                for (int j = startIndex; j < startIndex + CONSTRAINT; j++) {
                    int tidAndPid = tidAndPidsDevice[j];
                    if (tidAndPid == currentPoint.id)
                        break;

                    if (tidAndPid < 0 && atomicCAS(&tidAndPidsDevice[startIndex + j], -1, currentPoint.id) == -1)
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


/**
 * @brief Perform the GPU computation for finding proximity criteria.
 * @param pointArr Array of points.
 * @param numPoints Number of points.
 * @param actualTs Array of actual t values.
 * @param tidsAndPids 2D array of tids and pids.
 * @param numT Number of t values.
 * @param proximity Proximity value.
 * @param distance Distance value.
 * @param minTIndex Minimum t index.
 * @param maxTIndex Maximum t index.
 * @return 0 on success.
 */
int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int numT, int proximity, double distance, int minTIndex, int maxTIndex) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate memory on device for overall points buffer on device
    Point* pointsArrDevice = NULL;
    err = cudaMalloc((void**)&pointsArrDevice, numPoints * sizeof(Point));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy memory to device for overall points buffer on device
    err = cudaMemcpy(pointsArrDevice, pointArr, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory on device for actual t values buffer on device
    double* actualTsDevice = NULL;
    err = cudaMalloc((void**)&actualTsDevice, numT * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy memory to device for actual t values buffer on device
    err = cudaMemcpy(actualTsDevice, actualTs, numT * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the 2D array on the device
    int* tidsAndPidsDevice = NULL;
    err = cudaMalloc((void**)&tidsAndPidsDevice, numT * CONSTRAINT * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    intializeTidsAndPids <<< numT * CONSTRAINT, 1 >>>(tidsAndPidsDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Calculation of the efficient number of blocks to CUDA threads (256 threads per block)
    int numBlocks = ( numPoints * numT + THREADS_PER_BLOCK -1 )/ THREADS_PER_BLOCK;
    
    findProximityCriteria<<<numBlocks, THREADS_PER_BLOCK>>>(pointsArrDevice, numPoints, actualTsDevice, tidsAndPidsDevice, numT, proximity, distance, minTIndex, maxTIndex);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line** %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchronize to ensure all CUDA operations are completed
    cudaDeviceSynchronize();

    // Free device global memory
    err = cudaFree(pointsArrDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the final tid and pid array from the device to the host
    for (int i = 0; i < numT; i++){
        
        //Copying under each t index the ids of the Proximity Criteria points
        err =  cudaMemcpy(tidsAndPids[i],tidsAndPidsDevice + i * CONSTRAINT, CONSTRAINT * sizeof(int),cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // Free device global memory
    err = cudaFree(actualTsDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(tidsAndPidsDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reset the device
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
