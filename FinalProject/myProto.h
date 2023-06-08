#pragma once
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "point.h"

#define MASTER_PROC 0
#define CONSTRAINT 3
#define INPUT_FILE "input.txt"

void test(const Point* pointsArr, int numPoints, int* tOccurences, int tCount);

void readFromFile(Point* pointsArr, int* numPoints, int* tCount, int* proximity, double* radius);

int* sumProximityCriteria(const Point* pointsArr, int numPoints, int tCount, int proximity, double radius);

int computeOnGPU(const Point* pointsArr,  int numPoints, int tCount, int proximity, double radius);
