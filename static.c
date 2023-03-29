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
	loop = HEAVY * (6 % b);
	for (i = 0; i < loop; i++)
		sum += sin(a*exp(cos((double)(i%5))));
	return  sum;
}

int main(int argc, char **argv)
{
	int myid, numprocs, calcs = 0;
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
		for (int i = myid; i <ITER; i+=ITER/numprocs)
			sum += heavy(i,coef);

		printf("process %d calculated the sum %e\n",myid,sum);
		while (calcs < numprocs -1)
		{
			double res;
			MPI_Recv(&res, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			printf("process %d got the sum %e from process %d\n",myid,res,status.MPI_TAG);
			sum += res;
			calcs++;
		}
		printf("sum = %e\n", sum);
    }
	else {
		for (int i = myid; i <ITER; i+=ITER/numprocs)
			sum += heavy(i,coef);
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
		printf("process %d sent the sum %e to process 0\n",myid,sum);
	}
    MPI_Finalize();
}