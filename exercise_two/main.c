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

double f(double a, int b) {
	double value = 0;
	for (int i = 0; i < ITER; i++)
		value += sin(a * sin(b * cos(a * b)));
	return value;
}

double* calculateWithThreads(double *data, int numThreads, int pid) {

    int sizeData = sizeof(data) / sizeof(double);
    double *res = (double*) calloc(sizeData, sizeof(double));
    if (!res) {
        printf("Cannot allocate memory to result buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* This creates a team of threads; each thread has own copy of variables  */
#pragma omp parallel num_threads(numThreads)
    {
#pragma omp for
        for (int i = pid; i < sizeData; i++)
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
	int pid; /* rank of process */
	int numThreads; /* number of processes */
	MPI_Status status; /* return status for receive */

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numThreads);

	if (numThreads != 2) {
		printf("Please create only 2 processes");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (!pid) {
		int sizeData;
		FILE *filePtr = fopen("input.txt", "r");
		fread(&sizeData, sizeof(int), 1, filePtr);

		if(!sizeData){
			printf("Cannot read from file");
			MPI_Abort(MPI_COMM_WORLD,1);
			fclose(filePtr);
		}

		data = (double*) calloc(sizeData, sizeof(double));
		if (!data) {
			printf("Cannot allocate memory to data buffer");
			MPI_Abort(MPI_COMM_WORLD,1);
		}

		fread(data, sizeof(double), sizeData, filePtr);

		if(!data){
			printf("Cannot read from file");
			MPI_Abort(MPI_COMM_WORLD,1);
			fclose(filePtr);
		}
		fclose(filePtr);

		MPI_Send(data, sizeData, MPI_DOUBLE, SLAVE_PROC, 0, MPI_COMM_WORLD);

		res = calculateWithThreads(data, atoi(argv[1]), pid);
	

		MPI_Recv(data, sizeData, MPI_DOUBLE, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,
				&status);

		double sum = 0;
		for (int i = 0; i < sizeData; i++)
			sum += res[i] + data[i];

		printf("The total sum is:%f", sum);

		free(data);
		free(res);

	} else {

		MPI_Recv(data, 1, MPI_DOUBLE, SLAVE_PROC, MPI_ANY_TAG, MPI_COMM_WORLD,
				&status);

		res = calculateWithThreads(data, atoi(argv[1]), pid);

		MPI_Send(res, 1, MPI_DOUBLE, MASTER_PROC, 0, MPI_COMM_WORLD);

		free(res);
	}
}
