#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"



  __global__  void buildHist(int *h, int *temp, int dataSize,  int hist_per_thread) {
        int index = blockIdx.x * blockDim.x + threadIdx.x; // current thread index
        int offset= dataSize / NUM_BLOCKS / NUM_THREADS; // jumps of 2500

        // For each number and thread, go over all theard's histograms and find current num results
        for (int num = 0; num < RANGE; num++) {
            for (int hist_offset = 0; hist_offset < hist_per_thread; hist_offset++) {
                atomicAdd(&h[num], temp[offset_temp + (hist_offset * RANGE) + num]);
            } 
        }
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
    buildHist<<<  1,  RANGE>>>(d_h, d_temp);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the final histogram to the host
    err = cudaMemcpy(histValues, device_hist, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
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

