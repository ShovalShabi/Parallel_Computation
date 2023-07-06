#include <stddef.h> 
#include "myProto.h"
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

   Point* pointArr = NULL;
   int numPoints, tCount, proximity;
   int minTIndex, maxTIndex; //The start index and index to each process to check t values
   double distance;  //The recived distnace
   double* actualTs; //The actual values of the Ts
   int** tidsAndPids;
   int** allTidsAndPids;

   //Intiallizing MPI
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &size);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (size < 2) {
      perror("Run the executable with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   
   // Divide the tasks between both processes
   if (rank == 0)
      pointArr = readFromFile(&numPoints,&tCount,&proximity,&distance);

   // Broadcasting the total number of processes
   MPI_Bcast(&numPoints, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the tCount
   MPI_Bcast(&tCount, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the proximity count
   MPI_Bcast(&proximity, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the radius
   MPI_Bcast(&distance, 1, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);

   // printf("rank=%d N=%d tCount=%d K=%d, D=%lf\n",rank,numPoints,tCount,proximity,distance);

   // Define MPI_POINT datatypes
   MPI_Datatype MPI_POINT;
   MPI_Type_contiguous(sizeof(Point), MPI_BYTE, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);

   actualTs = (double*) malloc(sizeof(double) * tCount);

   if(!actualTs){
      perror("Allocating memory has been failed\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   if (!rank){
      buildTcountArr(actualTs,tCount); //Creating the array of the total Ts that needed to be calculted
   }

   //Alocating memory for the slave processes, is needed to acknowledge the whole buffer
   else {
      pointArr = (Point*) malloc(sizeof(Point) * numPoints);
      if(!pointArr){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
   }

   // Broadcasting all the actual t values
   MPI_Bcast(actualTs, tCount, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting all points
   MPI_Bcast(pointArr, numPoints, MPI_POINT, MASTER_PROC, MPI_COMM_WORLD);


   //The master creating the total array that holds all the tids and their Proximty Criteria points, will be recieved later
   int chunck = tCount/size;
   if (rank==0){
      numT = (tCount/size) + (tCount % size);
      allTidsAndPids = (int**) malloc(sizeof(int*) * tCount);

      if(!allTidsAndPids){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
      minTIndex = 0;
      maxTIndex = numT-1;
      for (int i =numT , proc=1; i < tCount ,proc < size; i+=chunck, proc++){
         MPI_Send(&i,1,MPI_INT,proc,0,MPI_COMM_WORLD); //Sending the minimum index of the actualTValues buffer to the process that need to calculate
         int maxTemp = i+chunck-1;
         MPI_Send(&maxTemp,1,MPI_INT,proc,0,MPI_COMM_WORLD); //Sending the maximum index of the actualTValues buffer to the process that need to calculate
      }
   }
   else{
      numT = chunck;
      MPI_Recv(&minTIndex,1,MPI_INT,MASTER_PROC,MPI_ANY_TAG,MPI_COMM_WORLD,&status);  //Recieving the minimum index of the actualTValues buffer to the process that need to calculate
      MPI_Recv(&maxTIndex,1,MPI_INT,MASTER_PROC,MPI_ANY_TAG,MPI_COMM_WORLD,&status);  //Recieving the maximum index of the actualTValues buffer to the process that need to calculate
   }

   printf("rank %d sweeps t values in range %d -- %d with chunk size of %d\n",rank,minTIndex,maxTIndex,numT);

   // The sub array that contains the pids of that apply Proximity Criteria under specific tid which is tidsAndPids[i]
   tidsAndPids = (int**) malloc(sizeof(int*) * tCount);

   if(!tidsAndPids){
      perror("Allocating memory has been failed\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   for (int i = 0; i < tCount; i++){
      tidsAndPids[i] = (int*) malloc (sizeof(int)*CONSTRAINT);

      if(!tidsAndPids){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
   }
   
   // Each process calculate its task on the GPU
   computeOnGPU(pointArr, numPoints, actualTs, tidsAndPids , tCount, proximity, distance, minTIndex, maxTIndex);

   // Collect the result from slave process
   if (rank == 0){
      allTidsAndPids = tidsAndPids; //The Criteria Points of the specified tids

      int** recvBuf = (int**) malloc(chunck*sizeof(int*));

      if (!chunck){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }

      for (int i = 1; i < tCount; i++){
         //Matching the relevant tids and the pid of the Criteria points of each process
         MPI_Recv(recvBuf,chunck,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status); //Reciving the other Criteria Points from the slave processes, the tag is the rank of the process that sent it
         
         for (int j = 0; j < chunck; j++){
            allTidsAndPids[status.MPI_TAG*chunck] = recvBuf[j];
         }
         
      }

      printf("\nFinished to calculate ProximityCriteria points.\n");

      for (int i = 0; i < tCount; i++)
      {
         for (int j = 0; j < CONSTRAINT; j++)
         {
            printf("tidsAndPids[%d][%d] = %d\t",i,j,allTidsAndPids[i][j]);
         }
         printf("\n");
         
      }
      

      writeToFile(OUTPUT_FILE, allTidsAndPids, actualTs, tCount);

      for (int i = 0; i < tCount; i++){
         free(allTidsAndPids[i]);
      }
      

      free(allTidsAndPids);  
   }

   // Send the data to master process
   else{
      MPI_Send(tidsAndPids+minTIndex,numT,MPI_INT,MASTER_PROC,rank,MPI_COMM_WORLD); //Sending the other Criteria Points from the slave processes, the tag is the rank of process that sent it
   }

   MPI_Type_free(&MPI_POINT);
   free(pointArr);

   for (int i = 0; i < tCount; i++){
      free(tidsAndPids[i]);
   }
   
   free(tidsAndPids);
   free(actualTs);

   MPI_Finalize();
   return 0;
}
