#pragma once
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MASTER_PROC 0
#define CONSTRAINT 3
#define THREADS_PER_BLOCK 256
#define INPUT_FILE "input.txt"
#define TEST_OUTPUT_FILE "testOutput.txt"


typedef struct {
    double x1;
    double x2;
    double a;
    double b;
    int id;
} Point;



Point* readFromFile(int* numPoints, int* tCount, int* proximity, double* radius);

void buildTcountArr(double* tArr, int tCount);

int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int numT, int proximity, double distance, int minTIndex, int maxTIndex);

void writeToFile(const char* fileName, int** tidsAndPids, double* actualTs, int tCount);
