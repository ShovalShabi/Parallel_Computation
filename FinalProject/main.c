#include <stddef.h> 
#include "myProto.h"
#include "point.h"
#include "mpi.h"

/**
 * @brief Main function for the MPI program
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return 0 on success
 */
int main(int argc, char *argv[]) {

   int size, rank, i;
   int numT;
   MPI_Status  status;

   Point* pointArr;
   int numPoints, tCount, proximity;
   double radius;
   int** tidsAndPids;
   int** allTidsAndPids;

   //Intiallizing MPI
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &size);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (size != 2) {
      perror("Run the executable with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   
   // Divide the tasks between both processes
   if (rank == 0)
      readFromFile(pointArr,&numPoints,&tCount,&proximity,&radius);

   // Broadcasting the total number of processes
   MPI_Bcast(&numPoints, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the tCount
   MPI_Bcast(&tCount, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the proximity count
   MPI_Bcast(&proximity, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the radius
   MPI_Bcast(&radius, 1, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);


   // Define MPI_POINT datatypes
   MPI_Datatype MPI_POINT;
   int blocklengths[5] = {1, 1, 1, 1, 1};
   MPI_Aint offsets[5];
   MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

   offsets[0] = offsetof(Point, x1);
   offsets[1] = offsetof(Point, x2);
   offsets[2] = offsetof(Point, a);
   offsets[3] = offsetof(Point, b);
   offsets[4] = offsetof(Point, id);

   MPI_Type_create_struct(numPoints, blocklengths, offsets, types, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);

   // Broadcasting all points
   MPI_Bcast(pointArr, numPoints, MPI_POINT, MASTER_PROC, MPI_COMM_WORLD);

   //The master creating the total array that holds all the tids and their Proximty Criteria points, will recive later by Gather 
   if (rank==0){
      numT = tCount/size + tCount % size;
      allTidsAndPids = (int**) malloc(sizeof(int*) * tCount);

      if(!allTidsAndPids){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
   }
   else{
      numT = tCount/size;
   }

   tidsAndPids = (int**) malloc(sizeof(int*) * numT);

   if(!tidsAndPids){
      perror("Allocating memory has been failed\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   // The sub array that contains the pids of that apply Proximity Criteria under specific tid which is tidsAndPids[i]
   for (int i = 0; i < numT; i++){
      tidsAndPids[i] = (int*) malloc(sizeof(int) * proximity);

      if(!tidsAndPids[i]){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
   }
   
   // Each process calculate its task on the GPU
   //-->computeOnGPU(data, DATA_SIZE/2, histValues);

   // Collect the result from slave process
   if (rank == 0){
      int expected = tCount/size;

      allTidsAndPids[0] = *tidsAndPids; //The Criteria Points of the specified tids

      for (int i = 1; i < size; i++){
         MPI_Recv(allTidsAndPids[i],expected,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status); //Reciving the other Criteria Points from the slave processes
      }

      //Write to output file here

      //test here

      free(allTidsAndPids);  
   }
   // Send the data to master process
   else{

   }

   MPI_Type_free(&MPI_POINT);
   free(pointArr);
   free(tidsAndPids);

   MPI_Finalize();
   return 0;
}
