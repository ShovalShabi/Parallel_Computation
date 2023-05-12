CC=mpicc
OBJS=main.o
ALL=main
DEBUG = -g
CFLAGS= -std=c99 -Wall -Werror $(DEBUG) -fopenmp

$(ALL):$(OBJS)
	$(CC) main.o -o main -lm -fopenmp
	
main.o: main.c


clean:
	rm -f $(OBJS) $(ALL) *.btr
