#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


__device__ double calculateDistance(const Point p1,const Point p2, double t){
    double xP1, yP1, xP2, yP2;

    xP1 = ((p1.x2 - p1.x1) / 2 ) * sin (t*M_PI /2) + (p1.x2 + p1.x1) / 2; 
    yP1 = p1.a*xP1 + p1.b;

    xP2 = ((p2.x2 - p2.x1) / 2 ) * sin (t*M_PI /2) + (p2.x2 + p2.x1) / 2; 
    yP2 = p2.a*xP2 + p2.b;


    return sqrt(pow(xP2-xP1,2) + pow(yP2-yP1,2));
}


__global__ void findProximityCriteria(Point* pointsArrDevice, int nCount, double* actualTsDevice, int** tidAndPidsDevice, int tCount, int proximity, double distance) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < nCount * tCount){
        int pid = threadId % nCount;  // The ID of the point
        int tid = threadId / nCount;  // The ID of the current t value

        int count = 0;
        for (int i = 0; i < nCount && i != pid; i++){
            double dist = calculateDistance(pointsArrDevice[pid], pointsArrDevice[i], actualTsDevice[tid]);

            if (dist <= distance)
                count++;
            
            if (count == proximity)
                break;  
        }

        if (count == proximity){
            for (int j = 0; j < CONSTRAINT; j++)
                if(!tidAndPidsDevice[tid][j])
                    atomicExch(&tidAndPidsDevice[tid][j],pid);
        }
    }
}



int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int numT, int proximity, double distance) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t pitch;

    // Allocate memory on device for overall points buffer on device
    Point* pointsArrDevice = NULL;
    err = cudaMalloc((void**)&pointsArrDevice, numPoints*sizeof(Point));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copying memory on device for overall points buffer on device
    err = cudaMemcpy(pointsArrDevice, pointArr, numPoints, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory on device for actual t values buffer on device
    double* actualTsDevice = NULL;
    err = cudaMalloc((void**) &actualTsDevice, numT*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copying memory on device for actual t values buffer on device
    err = cudaMemcpy(actualTsDevice, actualTs, numT, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the matching tids and pids to two dimensional array to the device
    int** tidsAndPidsDevice = NULL;
    err = cudaMallocPitch((void**) tidsAndPidsDevice, &pitch, numT * sizeof(int), CONSTRAINT);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copying the matching tids and pids to two dimensional array to the device
    err = cudaMemcpy2D(tidsAndPidsDevice, pitch, tidsAndPids, numT * sizeof(int), numT * sizeof(int) , CONSTRAINT, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int numBlocks = (int) ceil( numPoints * numT / THREADS_PER_BLOCK);

    // Finding all the Proximity Criteria of each distinct t value
    findProximityCriteria<<<numBlocks, THREADS_PER_BLOCK>>>(pointsArrDevice, numPoints, actualTsDevice, tidsAndPidsDevice, numT, proximity, distance);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchronize to ensure all CUDA operations are completed
    cudaDeviceSynchronize();

    // Copy the final histogram from the device to the host
    err = cudaMemcpy2D(tidsAndPids, numT * sizeof(int), tidsAndPidsDevice, pitch, numT * sizeof(int), CONSTRAINT, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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

    printf("Done\n");
    return 0;
}