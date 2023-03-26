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
	int myid, numprocs;
    MPI_Status status;
	double res_proc = 0, sum = 0;
	int currentNum = 0;

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
		while (currentNum < ITER)
		{
			printf("sending data %d\n",currentNum);
			MPI_Recv(NULL, 0 , MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == CONTINUE_TAG)
			{
				MPI_Send(&currentNum, 1 , MPI_INT, MPI_ANY_SOURCE, CONTINUE_TAG, MPI_COMM_WORLD); //Sending the data to the slave processes
				currentNum++;
			}

		}
		for (int i =1; i< numprocs; i++)
		{
			MPI_Send(NULL, 0 , MPI_INT, i, FINISH_TAG, MPI_COMM_WORLD); //Sending the data to the slave processes
		}
	}
	else
	{
		MPI_Send(&currentNum, 1 , MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD);
		while (status.MPI_TAG != FINISH_TAG)
		{
			MPI_Recv(&currentNum, 1 , MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			sum += heavy(currentNum,coef);
			MPI_Send(NULL, 0 , MPI_INT, 0, CONTINUE_TAG, MPI_COMM_WORLD); //Sending continue tag

		}
		MPI_Send(&sum, 1 , MPI_DOUBLE, 0, CONTINUE_TAG, MPI_COMM_WORLD);
	}

	if (myid == 0)
		printf("sum = %e\n", sum);	
    MPI_Finalize();
}