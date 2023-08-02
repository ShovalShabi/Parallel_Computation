#include "myProto.h"

/**
 * @brief Calculate the distance between two points at a given t value.
 * @param p1 The first point.
 * @param p2 The second point.
 * @param t The t value.
 * @return The calculated distance.
 */
double calculateDistance(const Point p1,const Point p2, double t){
    double xP1, yP1, xP2, yP2;

    xP1 = ((p1.x2 - p1.x1) / 2 ) * sin (t*M_PI /2) + (p1.x2 + p1.x1) / 2; 
    yP1 = p1.a*xP1 + p1.b;

    xP2 = ((p2.x2 - p2.x1) / 2 ) * sin (t*M_PI /2) + (p2.x2 + p2.x1) / 2; 
    yP2 = p2.a*xP2 + p2.b;


    return sqrt(pow(xP2-xP1,2) + pow(yP2-yP1,2));
}


/**
 * @brief Main function for the proximity criteria calculation program as sequential code.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 on success.
 */
int main(int argc, char *argv[]) {
    clock_t startTime, endTime;
    startTime = clock();
    
    Point *pointArr = NULL;
    int numPoints, tCount, proximity;
    int minTIndex, maxTIndex; // The start index and index to each process to check t values
    double distance;          // The received distance
    double *actualTs;         // The actual values of the Ts
    int** proximities;        // Array to store the proximity criteria
    
    // Read data from file
    pointArr = readFromFile(&numPoints, &tCount, &proximity, &distance);
    
    actualTs = (double *)malloc(sizeof(double) * tCount);
    int* proxCounter;
    
    if (!actualTs) {
        perror("Allocating memory has failed\n");
        exit(1);
    }
    
    // Build array of t values
    buildTcountArr(actualTs, tCount);
    
    proximities = (int**)malloc(sizeof(int*) * tCount);
    
    if (!proximities) {
        perror("Allocating memory has failed\n");
        exit(1);
    }
    
    // Initialize proximities array
    for (int i = 0; i < tCount; i++) {
        proximities[i] = (int*)malloc(sizeof(int) * CONSTRAINT);
        
        if (!proximities[i]) {
            perror("Allocating memory has failed\n");
            exit(1);
        }
        
        for (int j = 0; j < CONSTRAINT; j++) {
            proximities[i][j] = -1;
        }
    }
    
    proxCounter = (int*)calloc(tCount, sizeof(int));
    
    if (!proxCounter) {
        perror("Allocating memory has failed\n");
        exit(1);
    }
    
    int printed = 0, countPerT = 0;
    
    // Calculate proximity criteria
    for (int i = 0; i < tCount; i++) {
        for (int j = 0; j < numPoints; j++) {
            for (int k = 0; k < numPoints && j != k; k++) {
                if (calculateDistance(pointArr[j], pointArr[k], actualTs[i]) <= distance) {
                    countPerT++;
                }
            }
            
            if (countPerT == proximity && proxCounter[i] < CONSTRAINT) {
                for (int h = 0; h < CONSTRAINT; h++) {
                    if (pointArr[j].id == proximities[i][h]) {
                        break;
                    }
                    
                    if (proximities[i][h] < 0) {
                        proximities[i][h] = pointArr[j].id;
                        proxCounter[i]++;
                        break;
                    }
                }
            }
            
            if (proxCounter[i] == CONSTRAINT) {
                printed = 1;
            }
            
            countPerT = 0;
        }
    }
    
    printf("\nFinished calculating Proximity Criteria points.\n");

    // Write results to file
    writeToFile(TEST_OUTPUT_FILE, proximities, actualTs, tCount);

    // Free allocated memory
    free(proxCounter);
    free(pointArr);

    for (int i = 0; i < tCount; i++) {
        free(proximities[i]);
    }

    free(proximities);
    free(actualTs);

    endTime = clock();
    double res = ((double) endTime - startTime) / CLOCKS_PER_SEC;
    printf("\nTest passed successfully at %.4lf seconds!\n", res);

    printf("\nPlease open the created %s file to observe the results.\n\n", TEST_OUTPUT_FILE);

    return 0;
}
