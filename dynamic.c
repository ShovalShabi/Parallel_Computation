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
// its run time depends on a and b values.
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
	time_t start = time(NULL);
	time_t end;
	int myid, numprocs, currentNum = 0, currentProc = 1;
    MPI_Status status;
	double sum = 0;
	

	int coef = atoi(argv[1]);  //The conversion of the coefficient from string to integer

	/**************MPI setup****************/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	if (numprocs < MINIMUM_PROCS)//There is not enough processes to calculate heavy, 2 minimum are required
	{
		printf("Create at least 2 processes.\n");
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
	}  
		
	/*Master code block:
	**------------------
	**At first the master waits for initial response from the slave processes and then provides tasks dynamically.
	**When each slave processes is available, it'll send the current sum and will wait for the next number.
	**When all the calculations are done the master process sends the FINISH tag to the slave processes.
	**************************************************************************************************************/
    if (myid == 0) {
		while (currentProc < numprocs)
		{
			/*Recieving the result from a specific slave process that attached it's tag to the message*/
			double res;
			MPI_Recv(&res, 1 , MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			sum+=res;

			/*Sending the next number to specific slave process that attached it's tag to the message*/
			if (currentNum < ITER)
			{
				MPI_Send(&currentNum, 1, MPI_INT, status.MPI_TAG, CONTINUE_TAG, MPI_COMM_WORLD);
				currentNum++;
			}
			/*Sending Termination tag to specific slave processes*/
			else
			{
				MPI_Send(NULL, 0, MPI_CHAR, status.MPI_TAG, FINISH_TAG, MPI_COMM_WORLD);
				currentProc++;
			}
		}
		printf("sum = %e\n", sum);
		end = time(NULL);
		printf("The program runtime is %f secondes\n",difftime(end,start));
	}
	/*Slave code block:
	**-----------------
	**At first the slave sends an initial message with sum value of 0.
	**When the slave process recives the next number to caculate it'll send the result on the next itration.
	**When the slave process recives a FINISH tag, the slave will exit the while loop and will exit from the program as well.
	************************************************************************************************************************/
	else
	{
		while (status.MPI_TAG != FINISH_TAG)
		{
			MPI_Send(&sum, 1 , MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);  //Sending the result to the master process with process id as a tag
			MPI_Recv(&currentNum, 1 , MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);  //Recieving the next number to calculate
			sum = heavy(currentNum,coef);
		}
	}
    MPI_Finalize();
}