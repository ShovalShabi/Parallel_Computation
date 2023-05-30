#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"

#define DATA_SIZE 1000000
#define SLAVE_PROC 1
#define MASTER_PROC 0
#define MINIMUM_PROCS 2
#define MAX_VALUE 256
#define NUM_THREADS 20

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
 

int main(int argc, char *argv[]) {
   int size, rank, i;
   int *data;
   MPI_Status  status;

   srand(time(NULL));

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (size != 2) {
      perror("Run the example with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
   // Divide the tasks between both processes
   if (rank == 0) {

      // Allocate memory for the whole data array by running OMP threads for randomize values
      data = randomValues();
      MPI_Send(data + DATA_SIZE/2, DATA_SIZE/2 , MPI_INT, SLAVE_PROC, 0, MPI_COMM_WORLD);
   } else {

      MPI_Recv(data, DATA_SIZE/2, MPI_INT, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
   }

   int *histValues = (int*) calloc(DATA_SIZE, sizeof(int));
   if (!data) {
      perroer("Cannot allocate memory to result buffer");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   computeOnGPU(data,histValues,MAX_VALUE,NUM_THREADS);

   // Collect the result on one of processes
   if (rank == 0){
      int* histFromSlave;
      MPI_Recv(histValues, MAX_VALUE, MPI_INT, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      sumValues(histValues,histFromSlave);
      free(data);
   }
   else 
      MPI_Send(histValues, MAX_VALUE, MPI_INT, MASTER_PROC, 0, MPI_COMM_WORLD);

   free(histValues);
   MPI_Finalize();

   return 0;
}


