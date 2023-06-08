#include "myProto.h"
#include <mpi.h>

void test(Point* pointsArr, int numPoints, int* tOccurences, int tCount){

}

void readFromFile(Point* pointsArr, int* numPoints, int* tCount, int* proximity, double* radius){
    FILE *file;

    // Open the file for reading
    file = fopen(FILENAME_MAX, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return 1;
    }

    // Read the first line containing the configuration of the problem
    fscanf(file, "%d %d %f %d", numPoints, proximity, radius, tCount);

    pointsArr = (Point*) calloc(numPoints,sizeof(Point));

    // Read the N lines for point parameters
    for (int i = 0; i < numPoints; i++) {
        fscanf(file, "%d %f %f %f %f", &pointsArr[i].id, &pointsArr[i].x1, &pointsArr[i].x2, &pointsArr[i].a, &pointsArr[i].b);

        // Process the read parameters as needed
        // For example, print the values
        printf("ID: %d, x1: %f, x2: %f, a: %f, b: %f\n", pointsArr[i].id, pointsArr[i].x1, pointsArr[i].x2, pointsArr[i].a, pointsArr[i].b);
    }

    // Close the file
    fclose(file);
}

int* sumProximityCriteria(Point* pointsArr, int numPoints, int* tOccurences, int tCount){
    return 0;
}
