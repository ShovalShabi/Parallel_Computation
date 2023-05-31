#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"



  __global__  void buildHist(int* data, int dataSize, int* hist) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int chunck = (dataSize/(NUM_BLOCKS*NUM_THREADS));
    for (int i = index*chunck ; i < index*chunck+chunck; i++){
        atomicAdd(hist+data[i],1);
        printf("data -> %d\n",data[i]);
    }
    
  }


__global__  void initHist(int * h) {

  int index = threadIdx.x;
  h[index] = 0;

}

__global__ void check(int* h){
    int index = threadIdx.x;
    printf("hist[%d]=%d\n",index,h[index]);
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
    printf("Creating data array for device succeeded\n");

    int *device_hist = NULL;
    err = cudaMalloc((void **)&device_hist, RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Creating hitogram for device succeeded\n");

    // Copy the vector A to the device
    err = cudaMemcpy(device_data, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < DATA_SIZE; i++){
    //     printf("(%d)-->%d\n",i,device_data[i]);
    // }

    printf("Data copy from host to device succeeded\n");

    // Initialize vectors on device, 1 block and number of threads AS  RANGE
    initHist <<< 1 , RANGE >>> (device_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Initialization of device histogram succeeded\n");

    // Unify the results
    buildHist<<<  NUM_BLOCKS,  NUM_THREADS>>>(device_data,DATA_SIZE, device_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Calculting histogram succeeded\n");
    // for (int i = 0; i < RANGE; i++)
    // {
    //     printf("(%d)-->%d\n",i,device_hist[i]);
    // }
    
    // Copy the final histogram to the host
    check<<< 1, RANGE>>> (device_hist);

    err = cudaMemcpy(histValues, device_hist, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copying histogram succeeded\n");

    // Free device global memory
    err = cudaFree(device_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Free device data succeeded\n");

    err = cudaFree(device_hist);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Free histogram data succeeded\n");


    printf("Done\n");
    return 0;
}

