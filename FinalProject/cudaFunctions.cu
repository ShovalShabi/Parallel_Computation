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


__global__ void findProximityCriteria(Point* pointsArrDevice, int nCount, double* actualTsDevice, int** tidAndPidsDevice, int tCount, int proximity, double distance, int minTIndex, int maxTIndex) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < nCount * tCount){
        int indexPoint = threadId % nCount;  // The index of the point within the buffer
        int indexT = threadId / nCount;  // The index of the current t value

        //Making sure that the process caclute only the range of t values that assigned to it
        if ( minTIndex <= indexT <= maxTIndex  ){
            int count = 0;
            for (int i = 0; i < nCount && i != indexPoint; i++){
                double dist = calculateDistance(pointsArrDevice[indexPoint], pointsArrDevice[i], actualTsDevice[indexT]);

                if (dist <= distance)
                    count++;
                
                if (count == proximity)
                    break;  
            }

            if (count == proximity){
                for (int j = 0; j < CONSTRAINT; j++)
                    if(!tidAndPidsDevice[indexT][j])
                        atomicExch(&tidAndPidsDevice[indexT][j],pointsArrDevice[indexPoint].id);
            }
        }
    }
}

__global__ void intializeTidsAndPids(int* tidsAndPidsDevice){
    int threadId = threadIdx.x;

    tidsAndPidsDevice[threadId] = -1;
}



int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int tCount, int proximity, double distance, int minTIndex, int maxTIndex) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t pitch;

    // Allocate memory on device for overall points buffer on device
    Point* pointsArrDevice = NULL;
    err = cudaMalloc((void**)&pointsArrDevice, numPoints * sizeof(Point));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copying memory on device for overall points buffer on device
    err = cudaMemcpy(pointsArrDevice, pointArr, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory on device for actual t values buffer on device
    double* actualTsDevice = NULL;
    err = cudaMalloc((void**) &actualTsDevice, tCount * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copying memory on device for actual t values buffer on device
    err = cudaMemcpy(actualTsDevice, actualTs, tCount * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the matching tids and pids to two dimensional array to the device
    int** tidsAndPidsDevice = NULL;
    err = cudaMallocPitch((void**) tidsAndPidsDevice, &pitch, tCount * sizeof(int), CONSTRAINT);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copying the matching tids and pids to two dimensional array to the device
    err = cudaMemcpy2D(tidsAndPidsDevice, pitch, tidsAndPids, tCount * sizeof(int), tCount * sizeof(int) , CONSTRAINT, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    intializeTidsAndPids <<<1, tCount * CONSTRAINT>>>((int*) tidsAndPids);

    for (int i = 0; i < tCount; i++)
    {
        for (int j = 0; j < CONSTRAINT; j++)
        {
            printf("tidsAndPidsDevice[%d][%d] = %d",i,j,tidsAndPidsDevice[i][j]);
        }
        
    }
    

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

    // Copy the final histogram from the device to the host
    err = cudaMemcpy2D(tidsAndPids, tCount * sizeof(int), tidsAndPidsDevice, pitch, tCount * sizeof(int), CONSTRAINT, cudaMemcpyDeviceToHost);
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