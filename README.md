# Parallel_Computation - Assignment 1

The code is dealing with several Parallel Computation methodologies.

* *The static approach* -> each process gets clear work bounds and computes only the relevant tasks under its responsibility (The master usually gets tasks as well, and has an extra job to gather all the work)

* *The dynamic approach* -> each process gets a task dynamically, unrelated to the difficulty of the task.
Some will receive simple tasks and some might get a bit more difficult tasks (The master process won't get any task, only set up the data and gather the data from the slaves)

### How to Run The Code: ###
* There is Makefile supplied to you, all you need to do is yo open the terminal locally in the folder's path and then enter the command: `make`

* All the source files will immediately go through linkage by mpicc compiler

* There are **3 executables** *sequential*, *static*, and *dynamic*

* Choose the number of processes you would like to run, for example, the command: `mpiexec -np 5 ./static 10`, will run the static methodology with 5 processes with coefficient 10 as defined within the source file.

* To clean the file enter the command: `make clean`

**sequential executable is a regular executable, so please run regularly without mpiexec** 