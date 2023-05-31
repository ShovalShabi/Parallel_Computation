#include "myProto.h"
#include <mpi.h>



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
      data = (int*) calloc( DATA_SIZE/2 , sizeof(int));
      MPI_Recv(data, DATA_SIZE/2, MPI_INT, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
   }

   int *histValues = (int*) calloc(RANGE, sizeof(int));
   if (!histValues) {
      perror("Cannot allocate memory to result buffer");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   computeOnGPU(data, DATA_SIZE/2, histValues);

   // Collect the result on one of processes
   if (rank == 0){
      int* histFromSlave = (int*) calloc(RANGE ,sizeof(int));
      if (!histFromSlave) {
         perror("Cannot allocate memory to result buffer");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }

      MPI_Recv(histFromSlave, RANGE, MPI_INT, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      sumValues(histValues,histFromSlave);
      test(data, histValues);
   }
   else
      MPI_Send(histValues, RANGE, MPI_INT, MASTER_PROC, 0, MPI_COMM_WORLD);

   free(histValues);
   free(data);
   MPI_Finalize();
   return 0;
}
