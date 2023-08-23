#pragma once
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cstring>

#define MASTER_PROC 0
#define CONSTRAINT 3
#define THREADS_PER_BLOCK 256
#define INPUT_FILE "input.txt"
#define OUTPUT_FILE "output.txt"
#define TEST_OUTPUT_FILE "testOutput.txt"


typedef struct {
    double x1;
    double x2;
    double a;
    double b;
    int id;
} Point;



Point* readFromFile(int* numPoints, int* tCount, int* proximity, double* distance);

void buildTcountArr(double* tArr, int tCount);

//int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int numT, int proximity, double distance, int minTIndex, int maxTIndex);

int computeOnGPU(int N, int K, double D, int chunkSize, double *tValues, Point *allPoints, int *proximities);

void writeToFile(const char* fileName, int** tidsAndPids, double* actualTs, int tCount);
