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
    double xP1, yP1, xP2, yP2;

    xP1 = ((p1.x2 - p1.x1) / 2 ) * sin (t*M_PI /2) + (p1.x2 + p1.x1) / 2; 
    yP1 = p1.a*xP1 + p1.b;

    xP2 = ((p2.x2 - p2.x1) / 2 ) * sin (t*M_PI /2) + (p2.x2 + p2.x1) / 2; 
    yP2 = p2.a*xP2 + p2.b;

    return sqrt(pow(xP2-xP1,2) + pow(yP2-yP1,2));
}

/**
 * @brief Find the proximity criteria for each point and t value on the GPU.
 * @param pointsArrDevice Device array of points.
 * @param nCount Total number of points.
 * @param actualTsDevice Device array of actual t values.
 * @param tidAndPidsDevice Device array of tids and pids.
 * @param tCount Total number of t values.
 * @param proximity Proximity value.
 * @param distance Distance value.
 * @param minTIndex Minimum t index.
 * @param maxTIndex Maximum t index.
 */
__global__ void findProximityCriteria(Point* pointsArrDevice, int nCount, double* actualTsDevice, int* tidAndPidsDevice, int tCount, int proximity, double distance, int minTIndex, int maxTIndex) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < nCount * tCount){
        int indexPoint = threadId % nCount;  // The index of the point within the buffer
        int indexT = threadId / nCount;  // The index of the current t value

        //Making sure that the process calculates only the range of t values that are assigned to it
        if (minTIndex <= indexT <= maxTIndex){
            int count = 0;
            for (int i = 0; i < nCount && i != indexPoint; i++){
                double dist = calculateDistance(pointsArrDevice[indexPoint], pointsArrDevice[i], actualTsDevice[indexT]);

                if (dist <= distance)
                    count++;
                
                if (count == proximity)
                    break;  
            }

            //The exmined point reached K neighbors which is proximity values
            if (count == proximity){
                for (int j = 0; j < CONSTRAINT; j++){
                    if (tidAndPidsDevice[indexT * CONSTRAINT + j] == pointsArrDevice[indexPoint].id)
                        break;
                    
                    if(tidAndPidsDevice[indexT * CONSTRAINT + j] < 0){
                        atomicExch(&tidAndPidsDevice[indexT * CONSTRAINT + j],pointsArrDevice[indexPoint].id);
                        break;
                    }    
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
    int threadId = threadIdx.x;

    tidsAndPidsDevice[threadId] = -1;
}

/**
 * @brief Perform the GPU computation for finding proximity criteria.
 * @param pointArr Array of points.
 * @param numPoints Number of points.
 * @param actualTs Array of actual t values.
 * @param tidsAndPids 2D array of tids and pids.
 * @param tCount Number of t values.
 * @param proximity Proximity value.
 * @param distance Distance value.
 * @param minTIndex Minimum t index.
 * @param maxTIndex Maximum t index.
 * @return 0 on success.
 */
int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int tCount, int proximity, double distance, int minTIndex, int maxTIndex) {

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
    err = cudaMalloc((void**)&actualTsDevice, tCount * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy memory to device for actual t values buffer on device
    err = cudaMemcpy(actualTsDevice, actualTs, tCount * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the 2D array on the device
    int* tidsAndPidsDevice = NULL;
    err = cudaMalloc((void**)&tidsAndPidsDevice, tCount * CONSTRAINT * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    intializeTidsAndPids <<<1, tCount * CONSTRAINT>>>(tidsAndPidsDevice);

    
    //Calculation of the efficient number of blocks to CUDA threads (256 threads per block)
    int numBlocks = (int) ceil( numPoints * tCount / THREADS_PER_BLOCK);

    // Finding all the Proximity Criteria of each distinct t value
    findProximityCriteria<<<numBlocks, THREADS_PER_BLOCK>>>(pointsArrDevice, numPoints, actualTsDevice, tidsAndPidsDevice, tCount, proximity, distance, minTIndex, maxTIndex);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchronize to ensure all CUDA operations are completed
    cudaDeviceSynchronize();

    // Copy the final tid and pid array from the device to the host
    for (int i = minTIndex; i <= maxTIndex; i++){
        
        //Copying under each t index the ids of the Proximity Criteria points
        err =  cudaMemcpy(tidsAndPids[i],tidsAndPidsDevice + i * CONSTRAINT, CONSTRAINT * sizeof(int),cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // Free device global memory
    err = cudaFree(pointsArrDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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
