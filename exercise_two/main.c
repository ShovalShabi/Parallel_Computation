/*
 ============================================================================
 Name        : main.c
 Author      : Shoval Shabi 
 ID    		 : 208383885
 Description : A program that calcultes heavy function using OpenMP and MPI
 ============================================================================
 */
#include <omp.h>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ITER 1000000
#define SLAVE_PROC 1
#define MASTER_PROC 0
#define MINIMUM_PROCS 2

/*The heavy function*/
double f(double a, int b) {
	double value = 0;
	for (int i = 0; i < ITER; i++)
		value += sin(a * sin(b * cos(a * b)));
	return value;
}


double* calculateWithThreads(double *data, int numThreads,int numProcs ,int pid, int sizeData) {
	/*Allocating reslut buffer*/
    double *res = (double*) calloc(sizeData, sizeof(double));
    if (!res) {
        printf("Cannot allocate memory to result buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* This creates a team of threads, each thread has own copy of variables  */
	omp_set_num_threads(numThreads);
#pragma omp parallel
    {
#pragma omp for
		/* The data is calculated bu offsets assuming there is two processes and the current process has id of 0,
		** it will calculate data[0] data[2] data[4] and so on
		*/
        for (int i = pid; i < sizeData; i+=numProcs)
            res[i] = f(data[i], i);
    }
    return res;
}

/**
 * The main program, each process calculte the heavy function f, the first process is preparing the data and then calculates
 * the results of f function along with the second process
 */
int main(int argc, char *argv[]) {

	double *data = NULL, *res = NULL, sum = 0; /* Data array, result array and the total sum*/
	double startTime, endTime;
	int sizeData, pid , numProcs; /* Total variable count from input.txt, rank of process and number of process*/
	MPI_Status status; /* Return status for receive */

	/* Start up MPI */
	MPI_Init(&argc, &argv);

	/* Find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	/* Find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	startTime = MPI_Wtime(); // Start time

	if (numProcs != MINIMUM_PROCS) {
		perror("Please create only 2 processes\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (!pid) {
		/* This section handle with reading from file input.txt and allocating buffer for the data*/
		FILE *filePtr = fopen("input.txt", "r");

		if(!filePtr){
			perror("Cannot open file\n");
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
		}

		/* Reading the amount of variables */
		if(fscanf(filePtr,"%d",&sizeData) != 1){
			perror("Cannot read the number of data\n");
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
		}

		/* Allocating data buffer */
		data = (double*) calloc(sizeData, sizeof(double));

		if (!data) {
			perror("Cannot allocate memory to data buffer");
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		
		/* Reading each data variable */
		for(int i = 0; i<sizeData;i++){
			if(!fscanf(filePtr, "%lf", data+i)){
				perror("Cannot read data\n");
				MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
			}
		}

		fclose(filePtr);

		MPI_Send(&sizeData, 1, MPI_INT, SLAVE_PROC, 0, MPI_COMM_WORLD); // Sending the amount of data to the other process

		MPI_Send(data, sizeData, MPI_DOUBLE, SLAVE_PROC, 0, MPI_COMM_WORLD); // Sending the whole data buffer

	} else {

		MPI_Recv(&sizeData, 1, MPI_INT, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,&status); // Recieving the amount of data to the other process

		/* Allocating data buffer */
		data = (double*) calloc(sizeData, sizeof(double));
		if (!data) {
			perror("Cannot allocate memory to data buffer");
			MPI_Abort(MPI_COMM_WORLD,1);
		}

		MPI_Recv(data, sizeData, MPI_DOUBLE, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,&status); // Recieving the whole data buffer
	}

	/*Calculating the data by the helper method using OpenMP threads*/
	res = calculateWithThreads(data, atoi(argv[1]), numProcs, pid, sizeData);

	/*Summing the results of each process*/
	for (int i = 0; i < sizeData; i++)
		sum += res[i];
	
	if(!pid){
		double tempSum = 0;
		MPI_Recv(&tempSum, 1, MPI_DOUBLE, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, //Recieving the sum from the other process
				&status);
		sum+=tempSum;
		printf("The total sum is:%f\n", sum);
		endTime = MPI_Wtime(); // End time
		printf("The total time is %f seconds\n", endTime-startTime);
	}
	else{
		MPI_Send(&sum, 1, MPI_DOUBLE, MASTER_PROC, 0, MPI_COMM_WORLD); //Sending the sum from the master process
	}

	free(res);
	free(data);
	MPI_Finalize();
}