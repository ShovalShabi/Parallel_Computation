# Parallel_Computation - Assignment 2

The code is dealing with several Parallel Computation methodologies, both *OpenMp* threads and *MPI* processes.

* *The single host approach* -> The host will run two processes, one of the prcesses is reading and preparing the data from the file *input.txt* and then send it to the other process.
Each process calculates the heavy function using several *OpenMP* threads that user decided to create by mentioning the number of threads within the execute command.
When each process is finished to calculte the code it will sum the results, the slave process will send the sum to the master and then the reslut will bre presented.

* *The distributed host approach* -> The program will be excuted the same as the single host approach, but on different hosts on the same LAN network.

### How to Run The Code On Single Host: ###
* There is a Makefile supplied to you, all you need to do is to open the terminal locally under the folder's path and then enter the command: `make`

* All the source files will immediately go through linkage by mpicc compiler

* There is **1 executable** called *main*, and there is an input file called *input.txt*

* Please make sure that you select **only 2 processes**, you can choose the number of threads you would like to run, for example, the command:
`mpiexec -np 2 ./main 2`, will run the excutable main with 2 threads to each process

* To clean the workspace enter the command: `make clean`

### How to Run The Code On Several Hosts: ###
* The process to run the code is the same as the single host appraoch but now, you will have to make sure that there are two updated executables on both machines, you can compile them both using the command `make`

* All the source files will immediately go through linkage by mpicc compiler

* There will be  **2 executable** on seperated machines called *main*, each executable will use the same *input.txt* file, in addition there is a hostfile called *hosts.txt* for the distributed approach

* Please make sure that you select **only 2 processes** to each machine and you must add the flag `-hostfile` to the execute command, you can choose the number of threads you would like to run, for example, the command:
`mpiexec -hostfile hosts.txt -np 2 ./main 2`, will run two excutables, one on each machine, with 2 threads to each process

* To clean the workspace enter the command: `make clean` on both machines

**Before running the distributed approach make sure you putting the same updated files, and the correct ip addresses and the main host is written on top in file hosts.txt** 