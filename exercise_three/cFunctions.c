/**
 * @file myProto.c
 * @brief Functions for histogram calculation and testing
 */

#include "myProto.h"
#include <mpi.h>

/**
 * @brief Test function to compare calculated histogram with locally computed histogram
 * @param data Array of data elements
 * @param calculatedHist Array of calculated histogram values
 */
void test(int *data, int* calculatedHist) {
    int localHist[RANGE] ={0};

    for (int i = 0; i < DATA_SIZE; i++)
        localHist[data[i]]++;
    
    for (int i = 0; i < RANGE; i++){
        if (calculatedHist[i] != localHist[i]) {
           printf("Wrong Calculations - Failure of the test at data[%d]\n", i);
           return;
    	}
    }
    printf("The test passed successfully\n"); 
}

/**
 * @brief Generates an array of random values
 * @return Pointer to the array of random values
 */
int* randomValues() {
	/* Allocating data buffer */
    int *data = (int*) calloc(DATA_SIZE, sizeof(int));
    if (!data) {
        perror("Cannot allocate memory to result buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* This creates a team of threads, each thread has its own copy of variables */
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < DATA_SIZE; i++)
            data[i] = rand() % RANGE;
    }
    return data;
}

/**
 * @brief Sums the histogram values calculated by each process
 * @param hist Array of histogram values
 * @param slaveHist Array of histogram values from the slave process
 * @return Pointer to the updated histogram array
 */
int* sumValues(int* hist, int* slaveHist) {
    /* This creates a team of threads, each thread has its own copy of variables */
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < RANGE; i++)
            hist[i] +=  slaveHist[i];
    }
    return hist;
}
