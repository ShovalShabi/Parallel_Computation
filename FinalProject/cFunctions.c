#include "myProto.h"
#include <mpi.h>

// void test(Point* pointsArr, int numPoints, int* tOccurences, int tCount){

// }

void readFromFile(Point* pointsArr, int* numPoints, int* tCount, int* proximity, double* radius){
    FILE *file;

    // Open the file for reading
    file = fopen(INPUT_FILE, "r");
    if (file == NULL) {
        perror("Failed to open the file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    // Read the first line containing the configuration of the problem
    fscanf(file, "%d %d %lf %d", numPoints, proximity, radius, tCount);

    pointsArr = (Point*) calloc(*numPoints,sizeof(Point));

    if(!pointsArr){
        perror("Allocating memory has been failed\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    // Read the N lines for point parameters
    for (int i = 0; i < *numPoints; i++) {
        fscanf(file, "%d %lf %lf %lf %lf", &pointsArr[i].id, &pointsArr[i].x1, &pointsArr[i].x2, &pointsArr[i].a, &pointsArr[i].b);

        // Process the read parameters as needed
        // For example, print the values
        printf("ID: %d, x1: %lf, x2: %lf, a: %lf, b: %lf\n", pointsArr[i].id, pointsArr[i].x1, pointsArr[i].x2, pointsArr[i].a, pointsArr[i].b);
    }

    // Close the file
    fclose(file);
}

void buildTcountArr(double* tArr, int tCount){
    #pragma omp parallel for shared(tArr)
    for (int i = 0; i < tCount; i++){
        tArr[i] = 2 * i /tCount -1;
    }
    
}
