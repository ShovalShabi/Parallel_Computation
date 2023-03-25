CC=mpicc
OBJS=static.o dynamic.o sequencial.o
ALL=static dynamic
DEBUG = -g
CFLAGS= -std=c99 -Wall -Werror $(DEBUG)

$(ALL):$(OBJS)
	$(CC) static.o -o static -lm
	$(CC) dynamic.o -o dynamic -lm
	$(CC) sequencial.o -o dynamic -lm
	
static.o: static.c

dynamic.o: dynamic.c

sequencial.o: sequencial.c

clean:
	rm -f $(OBJS) $(ALL) 
