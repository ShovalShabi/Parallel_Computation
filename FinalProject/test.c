#include "myProto.h"

double calculateDistance(const Point p1,const Point p2, double t);

double calculateDistance(const Point p1,const Point p2, double t){
    double xP1, yP1, xP2, yP2;

    xP1 = ((p1.x2 - p1.x1) / 2 ) * sin (t*M_PI /2) + (p1.x2 + p1.x1) / 2; 
    yP1 = p1.a*xP1 + p1.b;

    xP2 = ((p2.x2 - p2.x1) / 2 ) * sin (t*M_PI /2) + (p2.x2 + p2.x1) / 2; 
    yP2 = p2.a*xP2 + p2.b;


    return sqrt(pow(xP2-xP1,2) + pow(yP2-yP1,2));
}
/**
 * @brief Main function for the MPI program
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return 0 on success
 */
int main(int argc, char *argv[])
{

    Point *pointArr = NULL;
    int numPoints, tCount, proximity;
    int minTIndex, maxTIndex; // The start index and index to each process to check t values
    double distance;          // The recived distnace
    double *actualTs;         // The actual values of the Ts
    int** proximities;

    pointArr = readFromFile(&numPoints, &tCount, &proximity, &distance);

    actualTs = (double *)malloc(sizeof(double) * tCount);
    
    int* proxCounter;

    if (!actualTs){
        perror("Allocating memory has been failed\n");
        exit(1);
    }
    
    buildTcountArr(actualTs, tCount); // Creating the array of the total Ts that needed to be calculted

    proximities = (int**) malloc (sizeof(int*) * tCount);
    if (!proximities){
        perror("Allocating memory has been failed\n");
        exit(1);
    }

    for (int i = 0; i < tCount; i++){
        proximities[i] = (int*) malloc(sizeof(int) * CONSTRAINT);

        if(!proximities[i]){
            perror("Allocating memory has been failed\n");
            exit(1);
        }

        for (int j = 0; j < CONSTRAINT; j++){
            proximities[i][j] = -1;
        }
    }

    proxCounter = (int*) calloc(tCount, sizeof(int));

    if (!proxCounter){
        perror("Allocating memory has been failed\n");
        exit(1);
    }
    
    int printed = 0, countPerT = 0;

    printf("here\n");

    for (int i = 0; i < tCount; i++){
        for (int j = 0; j < numPoints; j++){
            for (int k = 0; k < numPoints && j!=k; k++){
                if (calculateDistance(pointArr[j], pointArr[k], actualTs[i]) <= distance){
                    // printf("point %d is a proximity criteria at t =%lf\n",pointArr[j].id,actualTs[i]);
                    countPerT++;
                }
                
                if(countPerT == CONSTRAINT && proxCounter < CONSTRAINT){
                    proxCounter[i]++;
                    for (int h = 0; h < CONSTRAINT; h++){
                        if(proximities[i][h] < 0)
                            proximities[i][h] = pointArr[j].id;
                    }
                    break;
                }
            }
            if(proxCounter[i] == CONSTRAINT){
                printf("found %d ProximityCriteria points at t =%lf\n",CONSTRAINT,actualTs[i]);
                printed = 1;  
            }
            countPerT = 0;
        }

    }
    if (printed){
        for (int i = 0; i < tCount; i++){
            if(proxCounter[i] == CONSTRAINT){
                for (int j = 0; j < CONSTRAINT; j++){
                    printf("pointId=%d ",proximities[i][j]);
                }
                printf(" are ProximityCriteria at t=%lf\n",actualTs[i]);
            }            
        }
    }
    else{
        printf("There are no ProximtyCriteria Points!\n");
    }
    
    free(proxCounter);

    free(pointArr);

    for (int i = 0; i < tCount; i++){
        free(proximities[i]);
    }

    free(proximities);

    free(actualTs);

    return 0;
}
