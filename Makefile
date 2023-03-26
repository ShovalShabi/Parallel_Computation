CC=mpicc
OBJS=static.o dynamic.o sequential.o
ALL=static dynamic sequencial
DEBUG = -g
CFLAGS= -std=c99 -Wall -Werror $(DEBUG)

$(ALL):$(OBJS)
	$(CC) static.o -o static -lm
	$(CC) dynamic.o -o dynamic -lm
	$(CC) sequential.o -o sequencial -lm
	
static.o: static.c

dynamic.o: dynamic.c

sequential.o: sequential.c

clean:
	rm -f $(OBJS) $(ALL) *.btr
