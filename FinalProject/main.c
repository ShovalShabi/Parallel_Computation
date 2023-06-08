/**
 * @file main.c
 * @brief Example MPI code for histogram calculation using CUDA
 */

#include "myProto.h"
#include <mpi.h>

/**
 * @brief Main function for the MPI program
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return 0 on success
 */
int main(int argc, char *argv[]) {

   int size, rank, i;
   MPI_Status  status;

   Point* pointArr;
   int numPoints, tCount, proximity;
   double radius;

   //Intiallizing MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   // Define MPI_POINT datatypes
   MPI_Datatype MPI_POINT;
   MPI_Type_contiguous(sizeof(Point), MPI_BYTE, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);

   if (size != 2) {
      perror("Run the executable with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
   // Divide the tasks between both processes
   if (rank == 0) {

      readFromFile(pointArr,&numPoints,&tCount,&proximity,&radius);

      // Broadcasting the tCount
      MPI_Bcast(&tCount, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

      // Broadcasting the proximity count
      MPI_Bcast(&proximity, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

      // Broadcasting the radius
      MPI_Bcast(&radius, 1, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);

      int masterNumPoints = ceil(numPoints/size);
      int numPointsSlaves = numPoints/size;

      for (int i = 0; i < size; i++){
         
      }
      

   } else {
      // Allocate memory for partial data array to slave process
      data = (int*) calloc(DATA_SIZE/2 , sizeof(int));
      MPI_Recv(data, DATA_SIZE/2, MPI_INT, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
   }

   // Allocate memory for histogram
   int *histValues = (int*) calloc(RANGE, sizeof(int));
   if (!histValues) {
      perror("Cannot allocate memory to result buffer");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   // Each process calculate its task on the GPU
   computeOnGPU(data, DATA_SIZE/2, histValues);

   // Collect the result from slave process
   if (rank == 0){
      int* histFromSlave = (int*) calloc(RANGE ,sizeof(int));
      if (!histFromSlave) {
         perror("Cannot allocate memory to result buffer");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }

      MPI_Recv(histFromSlave, RANGE, MPI_INT, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      sumValues(histValues, histFromSlave);
      test(data, histValues);
   }
   // Send the data to master process
   else
      MPI_Send(histValues, RANGE, MPI_INT, MASTER_PROC, 0, MPI_COMM_WORLD);

   free(histValues);
   free(data);
   MPI_Finalize();
   return 0;
}
