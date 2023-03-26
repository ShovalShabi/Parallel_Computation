#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>

#define HEAVY 1000000
#define MINIMUM_PROCS 2
#define ITER 100
#define FINISH_TAG 1000

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

void clear(int* data, int* buf)
{
	free(data);
	free(buf);
}


int main(int argc, char **argv)
{
	int myid, numprocs, currentProc = 1;
    MPI_Status status;
	float res_proc = 0, sum = 0;
	int *data = NULL, *buf = NULL;

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
		data = (int*) malloc (sizeof(int)* ITER);  //Allocating memory for the data buffer
		if (!data)
		{
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
			clear(data,buf);
		}
			
		for (int i =0; i < ITER; i++)  //Setting the data to the slave processes
			data[i] = i;
			
		for (int i = 0; i < ITER; i+=ITER/(numprocs-1))  
		{
			printf("process %d sent numbers %d - %d to process %d\n",myid,i,i+ITER/(numprocs-1) -1 ,currentProc);
			MPI_Send(data + i, ITER/(numprocs-1) , MPI_INT, currentProc, 0, MPI_COMM_WORLD); //Sending the data to the slave processes
			currentProc++;
		}
		currentProc = 1;  //Setting current processes to process with ID of 1
    }
	else {
		buf = (int*) malloc(sizeof(int)*ITER/(numprocs-1)); //Allocating memory buffer for the recieved data
		if (!buf)
		{
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
			clear(data,buf);
		}
		MPI_Recv(buf, ITER/(numprocs-1) , MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		for (int i = 0; i<ITER/(numprocs-1) && buf + i; i++)
			res_proc += heavy(buf[i], coef);  //Executing the heacy function
			
		MPI_Send(&res_proc, 1, MPI_FLOAT, 0, FINISH_TAG, MPI_COMM_WORLD);
		free(buf);
	}

	while (myid == 0 && currentProc < numprocs)
	{
		MPI_Recv(&res_proc, 1, MPI_FLOAT, MPI_ANY_SOURCE, FINISH_TAG, MPI_COMM_WORLD, &status);
		sum+=res_proc;
		if (status.MPI_TAG == FINISH_TAG)
			currentProc++;
	}
	if (myid == 0)
	{
		printf("sum = %e\n", sum);
		free(data);
	}	
    MPI_Finalize();
}