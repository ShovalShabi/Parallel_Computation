/*
 ============================================================================
 Name        : Exercise2OpenMP.c
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
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

double f(double a, int b) {
	double value = 0;
	for (int i = 0; i < ITER; i++)
		value += sin(a * sin(b * cos(a * b)));
	return value;
}

double* calculateWithThreads(double *data, int numThreads,int numProcs ,int pid, int sizeData) {
    double *res = (double*) calloc(sizeData, sizeof(double));
    if (!res) {
        printf("Cannot allocate memory to result buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* This creates a team of threads; each thread has own copy of variables  */
	omp_set_num_threads(numThreads);
#pragma omp parallel
    {
#pragma omp for
        for (int i = pid; i < sizeData; i+=numProcs)
            res[i] = f(data[i], i);
    }
    return res;
}

/**
 * Hello OpenMP World prints the number of threads and the current thread id
 */
int main(int argc, char *argv[]) {

	double *data = NULL; /* data array */
	double *res = NULL; /* result array */
	int sizeData;
	int pid; /* rank of process */
	int numProcs; /* number of processes */
	MPI_Status status; /* return status for receive */

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	if (numProcs != MINIMUM_PROCS) {
		perror("Please create only 2 processes\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (!pid) {
		FILE *filePtr = fopen("input.txt", "r");

		if(!filePtr){
			perror("Cannot open file\n");
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
		}

		if(fscanf(filePtr,"%d",&sizeData) != 1){
			perror("Cannot read the number of data\n");
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
		}

		//printf("size buf %d\n",sizeData);
		data = (double*) calloc(sizeData, sizeof(double));
		if (!data) {
			printf("Cannot allocate memory to data buffer");
			MPI_Abort(MPI_COMM_WORLD,1);
		}
		
		for(int i = 0; i<sizeData;i++){
			if(!fscanf(filePtr, "%lf", data+i)){
				perror("Cannot read data\n");
				MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
			}
			//printf("the number %d is -> %f\n",i+1,data[i]);
		}

		fclose(filePtr);

		MPI_Send(&sizeData, 1, MPI_INT, SLAVE_PROC, 0, MPI_COMM_WORLD);

		MPI_Send(data, sizeData, MPI_DOUBLE, SLAVE_PROC, 0, MPI_COMM_WORLD);

		res = calculateWithThreads(data, atoi(argv[1]), numProcs, pid, sizeData);
	

		MPI_Recv(data, sizeData, MPI_DOUBLE, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,
				&status);

		printf("process pid=%d with result array:\n",pid);
		for(int i =0; i<sizeData; i++){
			printf("process %d with pos: %d, val->%f\n",pid,i,res[i]);
		}

		double sum = 0;
		for (int i = 0; i < sizeData; i++)
			sum += res[i] + data[i];

		printf("The total sum is:%f", sum);
	} else {

		MPI_Recv(&sizeData, 1, MPI_INT, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,&status);

		data = (double*) calloc(sizeData, sizeof(double));
		if (!data) {
			perror("Cannot allocate memory to data buffer");
			MPI_Abort(MPI_COMM_WORLD,1);
		}

		MPI_Recv(data, sizeData, MPI_DOUBLE, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,&status);

		res = calculateWithThreads(data, atoi(argv[1]), numProcs, pid, sizeData);

		printf("process pid=%d with result array:\n",pid);
		for(int i =0; i<sizeData; i++){
			printf("process %d with pos: %d, val->%f\n",pid,i,res[i]);
		}

		MPI_Send(res, sizeData, MPI_DOUBLE, MASTER_PROC, 0, MPI_COMM_WORLD);


	}
	free(res);
	free(data);
	MPI_Finalize();
}
