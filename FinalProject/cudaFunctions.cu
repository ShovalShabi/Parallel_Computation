#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * @brief CUDA kernel function to build histogram
 * @param data Input data array
 * @param dataSize Size of the input data array
 * @param hist Output histogram array
 */
__global__ void buildHist(int* data, int dataSize, int* hist) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int chunck = (dataSize / (NUM_BLOCKS * NUM_THREADS));
    
    for (int i = index * chunck; i < index * chunck + chunck; i++) {
        atomicAdd(&hist[data[i]], 1);
    }
}

/**
 * @brief CUDA kernel function to initialize histogram
 * @param h Histogram array to initialize
 */
__global__ void initHist(int* h) {
    int index = threadIdx.x;
    h[index] = 0;
}

/**
 * @brief Function to perform histogram calculation on the GPU
 * @param data Input data array
 * @param dataSize Size of the input data array
 * @param histValues Output histogram array
 * @return 0 on success
 */
int computeOnGPU(int* data, int dataSize, int* histValues) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t size = dataSize * sizeof(int);

    // Allocate memory on device for overall data buffer
    int* device_data = NULL;
    err = cudaMalloc((void**)&device_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory on device for histogram buffer
    int* device_hist = NULL;
    err = cudaMalloc((void**)&device_hist, RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the input data to the device
    err = cudaMemcpy(device_data, data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize histogram on the device, one block and threads as the amount of RANGE
    initHist<<<1, RANGE>>>(device_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Build the histogram on the device, blocks as the amount of NUM_BLOCKS and threads as the amount of RANGE
    buildHist<<<NUM_BLOCKS, NUM_THREADS>>>(device_data, dataSize, device_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchronize to ensure all CUDA operations are completed
    cudaDeviceSynchronize();

    // Copy the final histogram from the device to the host
    err = cudaMemcpy(histValues, device_hist, sizeof(int) * RANGE, cudaMemcpyDeviceToHost);
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

    // Free device global memory
    err = cudaFree(device_hist);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}