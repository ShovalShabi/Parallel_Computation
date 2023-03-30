#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define HEAVY 1000000
#define ITER 100

// This function performs heavy computations,
// its run time depends on a and b values
// DO NOT change this function
double heavy(int a, int b)
{
	int i, loop;
	double sum = 0;
	loop = HEAVY * (rand() % b);
	for (i = 0; i < loop; i++)
		sum += sin(a * exp(cos((double)(i % 5))));
	return sum;
}

// Sequential code to be parallelized
int main(int argc, char **argv)
{
	time_t start = time(NULL);
	time_t end;
	int coef = atoi(argv[1]); // The conversion of the coefficient from string to integer
	double sum = 0;
	for (int i = 0; i < ITER; i++)
		sum += heavy(i, coef);
	printf("sum = %e\n", sum);
	end = time(NULL);
	printf("The program runtime is %f secondes\n", difftime(end, start));
}