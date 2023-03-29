#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>

#define HEAVY 1000000
#define MINIMUM_PROCS 2
#define ITER 100
#define FINISH_TAG 1000
#define CONTINUE_TAG 100

// This function performs heavy computations, 
// its run time depends on a and b values
// DO NOT change this function
double heavy(int a, int b) {
	int i, loop;
	double sum = 0;
	loop = HEAVY * (rand() % b);
	for (i = 0; i < loop; i++)
		sum += sin(a*exp(cos((double)(i%5))));
	return  sum;
}

int main(int argc, char **argv)
{
	int myid, numprocs, currentNum = 0;
    MPI_Status status;
	double sum = 0;
	

	int coef = atoi(argv[1]);  //The coefficient of the the number

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	if (numprocs < MINIMUM_PROCS)//There is not enough processes to calculate heavy, 2 minimum are required
	{
		printf("Create at least 2 processes.\n");
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
	}  
		

    if (myid == 0) {
		for (int i = 1; i < numprocs; i++)
			MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //Recieving ready signal of all slave processes

		while (currentNum < ITER)
		{
			double res;
			MPI_Recv(NULL, 0 , MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //Recieving the result from a specific slave process that attached it's tag to the message
			MPI_Send(&currentNum, 1, MPI_INT, status.MPI_TAG, CONTINUE_TAG, MPI_COMM_WORLD); //Sending the next number to specific slave process that attached it's tag to the message
			MPI_Recv(&res, 1 , MPI_DOUBLE, status.MPI_TAG, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //Recieving the result from a specific slave process that attached it's tag to the message
			sum+=res;
			currentNum++;
		}
		// for (int i = 0; i < ITER; i++)
		// {
		// 	MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //Recieving ready signal of all slave processes
			
			

		// }

		for (int i = 1; i < numprocs; i++)
			MPI_Send(&i, 0, MPI_CHAR, i, FINISH_TAG, MPI_COMM_WORLD); //Sending Termination tag to all slave processes
		printf("sum = %e\n", sum);
	}
	else
	{
		// MPI_Send(NULL, 0 , MPI_CHAR, 0, myid, MPI_COMM_WORLD);  //Sending ready signal to master process
		while (status.MPI_TAG != FINISH_TAG)
		{
			MPI_Send(NULL, 0 , MPI_CHAR, 0, myid, MPI_COMM_WORLD);  //Sending ready signal to master process
			MPI_Recv(&currentNum, 1 , MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);  //Recieving the next number to calculate
			sum = heavy(currentNum,coef);
			MPI_Send(&sum, 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);  //Sending continue tag
		}
	}
    MPI_Finalize();
}