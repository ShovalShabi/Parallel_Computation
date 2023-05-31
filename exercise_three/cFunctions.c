#include "myProto.h"
#include <stdio.h>

#define MAX_VALUE 256

void test(int *data, int* calculatedHist) {
    int localHist[MAX_VALUE] ={0};

    for (int i = 0; i < DATA_SIZE; i++)
        localHist[data[i]]++;
    
    for (int i = 0; i < MAX_VALUE; i++){
        if (calculatedHist[i] != localHist[i]) {
           printf("Wrong Calculations - Failure of the test at data[%d]\n", i);
           return;
    	}
    }
    printf("The test passed successfully\n"); 
}


/*
Simple MPI+OpenMP+CUDA Integration example
Initially the array of size 4*PART is known for the process 0.
It sends the half of the array to the process 1.
Both processes start to increment members of thier members by 1 - partially with OpenMP, partially with CUDA
The results is send from the process 1 to the process 0, which perform the test to verify that the integration worked properly
*/
int* randomValues() {
	/*Allocating data buffer*/
    int *data = (int*) calloc(DATA_SIZE, sizeof(int));
    if (!data) {
        perroer("Cannot allocate memory to result buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* This creates a team of threads, each thread has own copy of variables  */
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
#pragma omp for
		/* The data is calculated bu offsets assuming there is two processes and the current process has id of 0,
		** it will calculate data[0] data[2] data[4] and so on
		*/
        for (int i = 0; i < DATA_SIZE; i++)
            data[i] = rand() % MAX_VALUE;
    }
    return data;
}


/*

*/
int* sumValues(int* hist, int* slaveHist) {
    /* This creates a team of threads, each thread has own copy of variables  */
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
#pragma omp for
		/* The data is calculated bu offsets assuming there is two processes and the current process has id of 0,
		** it will calculate data[0] data[2] data[4] and so on
		*/
        for (int i = 0; i < DATA_SIZE; i++)
            hist[i] +=  slaveHist[i];
    }
    return hist;
}
 