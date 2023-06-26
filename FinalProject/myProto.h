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


typedef struct {
    double x1;
    double x2;
    double a;
    double b;
    int id;
} Point;


// void test(const Point* pointsArr, int numPoints, int* tOccurences, int tCount);

void readFromFile(Point* pointsArr, int* numPoints, int* tCount, int* proximity, double* radius);

void writeToFile(int** alltidAndPids, int tCount, double* actualTs);

void buildTcounrArr(double* tArr, int tCount);

int computeOnGPU(Point* pointArr, int numPoints, double* actualTs, int** tidsAndPids , int numT, int proximity, double distance);
