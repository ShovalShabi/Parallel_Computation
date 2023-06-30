NUM_PROCESSES = 2

build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o -lm
	mpicxx -fopenmp -o ProximityCriteria  main.o cFunctions.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	mpicxx -c test.c -o test.o
	mpicxx -fopenmp -o Test test.o cFunctions.o
	

clean:
	rm -f *.o ProximityCriteria *.btr Test

run:
	mpiexec -np $(NUM_PROCESSES) ./ProximityCriteria

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./ProximityCriteria

runTest:
	./Test