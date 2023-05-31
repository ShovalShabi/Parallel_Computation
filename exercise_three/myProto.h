#pragma once
#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define DATA_SIZE 1000000
#define SLAVE_PROC 1
#define MASTER_PROC 0
#define MINIMUM_PROCS 2
#define RANGE 256
#define NUM_THREADS 20
#define NUM_BLOCKS 10


void test(int *data, int* calculatedHist);
int* randomValues();
int* sumValues(int* hist, int* slaveHist);
int computeOnGPU(int *data, int dataSize ,int* histValues);
