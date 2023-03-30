#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>

#define HEAVY 1000000
#define MINIMUM_PROCS 2
#define ITER 100

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
	time_t start = time(NULL);
	time_t end;
	int myid, numprocs, calcs = 0;
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
	**At first the master calculate the numbers that are needed to be calculated, and only then it will sum the results
	**from the slave processes.
	**The numbers are calculted from the current id number of the process with an offset of ITER/numprocs.
	******************************************************************************************************************/
    if (myid == 0)
	{
		for (int i = myid; i <ITER; i+=ITER/numprocs)
			sum += heavy(i,coef);

		while (calcs < numprocs -1)
		{
			double res;
			MPI_Recv(&res, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			sum += res;
			calcs++;
		}
		printf("sum = %e\n", sum);
		end = time(NULL);
		printf("The program runtime is %f secondes\n",difftime(end,start));
    }
	/*Slave code block:
	**-----------------
	**The slave process caculates all the number that are needed to be calculated from current id number of the process with
	**an offset of ITER/numprocs.
	**When the slave process is finished, it'll send the toatl result of its entire calculations to the master process.
	************************************************************************************************************************/
	else {
		for (int i = myid; i <ITER; i+=ITER/numprocs)
			sum += heavy(i,coef);
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
	}
    MPI_Finalize();
}