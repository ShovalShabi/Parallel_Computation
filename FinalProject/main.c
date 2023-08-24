#include <stddef.h> 
#include "myProto.h"
#include "mpi.h"

/**
 * @brief Main function for the MPI program, from this file the MPI processes are being created.
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return 0 on success
 */
int main(int argc, char *argv[]) {

   clock_t startTime, endTime;
   startTime = clock();

   int size, rank, numT;
   MPI_Status  status;

   Point* pointArr = NULL;
   int numPoints, tCount, proximity;
   int minTIndex, maxTIndex; // The start index and index to each process to check t values
   double distance;  // The recieved distnace
   double* actualTs; // The actual values of the Ts
   int* allTidsAndPids; // A two dimensional array that holds the relevant ids of point who correspond with ProximityCriteria (exposed to master only)
   int* tidsAndPids;  // The same as above, this pointer is independent to each process


   // Intiallizing MPI
   MPI_Init(&argc, &argv);

   // Retrieving the sizeof MPI
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   // Giving id to weach process 
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (size < 2) {
      perror("Run the executable with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   
   // Master handling input file
   if (rank == 0){
      pointArr = readFromFile(&numPoints,&tCount,&proximity,&distance);

      if (size >= tCount){  //Tcount is measured from 0 to tCount which is actually grater by one of what written in input.txt
         perror("Cannot run more processes than actual number of tCount\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }

   }

   // Broadcasting the total number of processes
   MPI_Bcast(&numPoints, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the tCount
   MPI_Bcast(&tCount, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the proximity count
   MPI_Bcast(&proximity, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

   // Broadcasting the radius
   MPI_Bcast(&distance, 1, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);


   // Define MPI_POINT datatypes
   MPI_Datatype MPI_POINT;
   MPI_Type_contiguous(sizeof(Point), MPI_BYTE, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);



   // Building the array of t values that needed to be calculated By the master process
   if (!rank){
      // An array of t values
      actualTs = (double*) malloc(sizeof(double) * tCount);

      if(!actualTs){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }

      buildTcountArr(actualTs,tCount); //Creating the array of the total Ts that needed to be calculted
   }

   // Alocating memory for the slave processes, is needed to acknowledge the whole buffer
   else {
      pointArr = (Point*) malloc(sizeof(Point) * numPoints);
      if(!pointArr){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
   }

   // Broadcasting all points
   MPI_Bcast(pointArr, numPoints, MPI_POINT, MASTER_PROC, MPI_COMM_WORLD);


   int chunck = tCount/size; // The chunk size of each process except master
   
   // The master creating the total array that holds all the tids and their Proximty Criteria points, will be recieved later
   if (rank==0){
      numT = (tCount/size) + (tCount % size);

      minTIndex = 0;
      maxTIndex = numT-1;
      for (int i =numT , proc=1; i < tCount ,proc < size; i+=chunck, proc++){
         MPI_Send(&i,1,MPI_INT,proc,0,MPI_COMM_WORLD); //Sending the minimum index of the actualTValues buffer to the process that need to calculate
         int maxTemp = i+chunck-1;
         MPI_Send(&maxTemp,1,MPI_INT,proc,0,MPI_COMM_WORLD); //Sending the maximum index of the actualTValues buffer to the process that need to calculate
         MPI_Send(actualTs + i,chunck,MPI_DOUBLE,proc,0,MPI_COMM_WORLD); //Sending the maximum index of the actualTValues buffer to the process that need to calculate
      }
   }
   else{
      numT = chunck;
      MPI_Recv(&minTIndex,1,MPI_INT,MASTER_PROC,MPI_ANY_TAG,MPI_COMM_WORLD,&status);  //Recieving the minimum index of the actualTValues buffer to the process that need to calculate
      MPI_Recv(&maxTIndex,1,MPI_INT,MASTER_PROC,MPI_ANY_TAG,MPI_COMM_WORLD,&status);  //Recieving the maximum index of the actualTValues buffer to the process that need to calculate
      
      // An array of t values
      actualTs = (double*) malloc(sizeof(double) * numT);

      if(!actualTs){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }
      MPI_Recv(actualTs,numT,MPI_DOUBLE,MASTER_PROC,MPI_ANY_TAG,MPI_COMM_WORLD,&status);  //Recieving a chunck of the actual t values
   }

   // The independent array that contains the pids of that apply ProximityCriteria under specific tid which is tidsAndPids[i] in size of CONSTRAINT
   tidsAndPids = (int*) malloc(sizeof(int) * numT * CONSTRAINT);

   if(!tidsAndPids){
      perror("Allocating memory has been failed\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   //Initializing the tidsAndPids array for further calculation
   initiallizeTidsAndPids(tidsAndPids, numT);

 
   // Each process calculate its task on the GPU
   // computeOnGPU(pointArr, numPoints, actualTs, tidsAndPids , numT, proximity, distance, minTIndex, maxTIndex);
   computeOnGPU(numPoints, proximity, distance, numT, actualTs, pointArr, tidsAndPids);


   // Collect the result from slave process
   if (rank == 0){
      allTidsAndPids = (int*) malloc(sizeof(int) * tCount * CONSTRAINT);

      if(!allTidsAndPids){
         perror("Allocating memory has been failed\n");
         MPI_Abort(MPI_COMM_WORLD, __LINE__);
      }

      //The CriteriaPoints of the specified tids of the master
      for (int i = 0; i <= numT * CONSTRAINT; i++){
         allTidsAndPids[i] = tidsAndPids[i];
      }

      // Recieving the ids under specific t index
      for (int i = maxTIndex + 1; i < tCount; i+=chunck){
         //Matching the relevant tids and the pid of the Criteria points of each process
         MPI_Recv(allTidsAndPids+i*CONSTRAINT,CONSTRAINT*chunck,MPI_INT,MPI_ANY_SOURCE,i,MPI_COMM_WORLD,&status); //Reciving the other Criteria Points from the slave processes, the tag is the rank of the process that sent it  
         

      }

      printf("\nFinished to calculate ProximityCriteria points.\n");

      //Writing the data to an output file
      writeToFile(OUTPUT_FILE, allTidsAndPids, actualTs, tCount);

      printf("\nPlease open the created %s file to observe the results.\n",OUTPUT_FILE);

      free(allTidsAndPids);

      endTime = clock();
      double res = ((double) endTime - startTime) / CLOCKS_PER_SEC;
      printf("\nProgram finished caclulation successfully at %.4lf seconds!\n\n",res);  
   }

   // Send the data to master process
   else{
      
      MPI_Send(tidsAndPids,CONSTRAINT * numT,MPI_INT,MASTER_PROC,minTIndex,MPI_COMM_WORLD); //Sending the other Criteria Points from the slave processes, the tag is the current index that beinfg requested by the master process
      free(tidsAndPids);
      
   }

   MPI_Type_free(&MPI_POINT);
   free(pointArr);
   free(actualTs);

   MPI_Finalize();
   return 0;
}
