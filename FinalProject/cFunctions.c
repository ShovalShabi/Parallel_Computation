#include "myProto.h"
#include "mpi.h"


Point* readFromFile(int* numPoints, int* tCount, int* proximity, double* radius){
    FILE *file;
    Point* pointsArr =NULL;

    // Open the file for reading
    file = fopen(INPUT_FILE, "r");
    if (file == NULL) {
        perror("Failed to open the file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    // Read the first line containing the configuration of the problem
    if (fscanf(file, "%d %d %lf %d", numPoints, proximity, radius, tCount) != 4){
        perror("Failed to to read from file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
        fclose(file);
    }
    *tCount+=1; // Including the last point of tCount

    pointsArr = (Point*) malloc((*numPoints)*sizeof(Point));

    if(!pointsArr){
        perror("Allocating memory has been failed\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    // Read the N lines for point parameters
    for (int i = 0; i < *numPoints; i++) {
        if(fscanf(file, "%d %lf %lf %lf %lf", &pointsArr[i].id, &pointsArr[i].x1, &pointsArr[i].x2, &pointsArr[i].a, &pointsArr[i].b)!=5){
            perror("Failed to to read from file.\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
            fclose(file);
            free(pointsArr);
        }

        // Process the read parameters as needed
        // For example, print the values
        printf("ID: %d, x1: %lf, x2: %lf, a: %lf, b: %lf\n", pointsArr[i].id, pointsArr[i].x1, pointsArr[i].x2, pointsArr[i].a, pointsArr[i].b);
    }

    // Close the file
    fclose(file);

    printf("\nRead %s file successfully.\n",INPUT_FILE);
    return pointsArr;
}

void buildTcountArr(double* tArr, int tCount){
    #pragma omp parallel for shared(tArr)
    for (int i = 0; i < tCount; i++){
        tArr[i] = 2 * i / (double)tCount -1;
        // int tid = omp_get_thread_num();
        // printf("Thread %d caclulated t=%lf with t=%d\n",tid,tArr[i],i);
    }
    printf("finished caclulate all t values.\n");
    
}
