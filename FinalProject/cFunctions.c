#include "myProto.h"
#include "mpi.h"


Point* readFromFile(int* numPoints, int* tCount, int* proximity, double* radius){
    FILE *file;
    Point* pointsArr =NULL;

    // Open the file for reading
    file = fopen(INPUT_FILE, "r");
    if (file == NULL) {
        perror("Failed to open the file.\n");
        exit(1);
    }

    // Read the first line containing the configuration of the problem
    if (fscanf(file, "%d %d %lf %d", numPoints, proximity, radius, tCount) != 4){
        perror("Failed to to read from file.\n");
        fclose(file);
        exit(1);
    }
    *tCount+=1; // Including the last point of tCount

    pointsArr = (Point*) malloc((*numPoints)*sizeof(Point));

    if(!pointsArr){
        perror("Allocating memory has been failed\n");
        exit(1);
    }

    // Read the N lines for point parameters
    for (int i = 0; i < *numPoints; i++) {
        if(fscanf(file, "%d %lf %lf %lf %lf", &pointsArr[i].id, &pointsArr[i].x1, &pointsArr[i].x2, &pointsArr[i].a, &pointsArr[i].b)!=5){
            perror("Failed to to read from file.\n");
            fclose(file);
            free(pointsArr);
            exit(1);
        }

        // Process the read parameters as needed
        // For example, print the values
        // printf("ID: %d, x1: %lf, x2: %lf, a: %lf, b: %lf\n", pointsArr[i].id, pointsArr[i].x1, pointsArr[i].x2, pointsArr[i].a, pointsArr[i].b);
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
    printf("\nfinished caclulate all t values.\n");
    
}


void writeToFile(const char* fileName, int** tidsAndPids, double* actualTs, int tCount){
    int printed = 0;
    FILE* file;

    file = fopen(fileName,"w");

    if(!file){
        perror("Failed to write to file\n");
        exit(1);
    }

    int* proxCounter = (int*) calloc(tCount,sizeof(int));
    
    if(!proxCounter){
        perror("Allocating memory has been failed\n");
        fclose(file);
        exit(1);
    }

    #pragma omp parallel for shared(proxCounter)
    for (int i = 0; i < tCount; i++){
        for (int j = 0; j < CONSTRAINT; j++){
            if (tidsAndPids[i][j] > 0)
                proxCounter[i]++;
        }
        
    }
    for (int i = 0; i < tCount; i++){   
        if (proxCounter[i] == CONSTRAINT){
            printed = 1;
            break;
        }
    }
    
    
    if (printed){
        for (int i = 0; i < tCount; i++){
            if(proxCounter[i] == CONSTRAINT){
                for (int j = 0; j < CONSTRAINT; j++){
                    fprintf(file,"pointId=%d ",tidsAndPids[i][j]);
                }
                fprintf(file," are ProximityCriteria at t=%lf\n",actualTs[i]);
            }            
        }
    }
    else
        fprintf(file,"There are no ProximtyCriteria Points!\n");
    
    free(proxCounter);
    fclose(file);
}
