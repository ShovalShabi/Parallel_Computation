#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"



  __global__  void buildHist(int* data, int dataSize, int* hist) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int chunck = (dataSize/(NUM_BLOCKS*NUM_THREADS));
    for (int i = index*chunck ; i < index*chunck+chunck; i++)
        atomicAdd(&hist[data[i]],1);
    
  }


__global__  void initHist(int * h) {

  int index = threadIdx.x;
  h[index] = 0;

}

int computeOnGPU(int *data, int dataSize ,int* histValues) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t size = dataSize * sizeof(int);

    // Allocate memory on device
    int *device_data = NULL;
    err = cudaMalloc((void **)&device_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *device_hist = NULL;
    err = cudaMalloc((void **)&device_hist, RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the vector A to the device
    err = cudaMemcpy(device_data, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Initialize vectors on device, 1 block and number of threads AS  RANGE
    initHist <<< 1 , RANGE >>> (device_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Unify the results
    buildHist<<<  NUM_BLOCKS,  NUM_THREADS>>>(device_data,dataSize, device_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Copy the final histogram to the host
    err = cudaMemcpy(histValues,device_hist,sizeof(int)*RANGE,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(device_hist);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

