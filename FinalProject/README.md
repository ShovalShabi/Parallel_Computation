# Parallel_Computation - Final Project

The code is dealing with several Parallel Computation methodologies, *CUDA* *OpenMp* threads and *MPI* processes.

* *The single host approach* -> The host will run sevral processes, one of the prcesses is reading and preparing the data from file *input.txt*, and assign the values of **N** (number of points), **K** (number of neighbours), **D** (distance from exmined point to its neighbours) and **tCount** (the number of t values)..
Each process calculates the Proximity Criteria problem by operating CUDA threads on the GPU accorfing to the data that has been configured.
When each process is finished to calculate the Proximty Criteria to it's tValues indexes, the slave process will send the data to the master and then the master will add up the buffers that it recieves and then present it in output file called *output.txt*.

* *The distributed host approach* -> The program will be excuted the same as the single host approach, but on different hosts on the same LAN network, the number of processes in this case is limited to two processes.

### How to Run The Code On Single Host: ###
* There is a Makefile supplied to you, all you need to do is to open the terminal locally under the folder's path and then enter the command: `make`

* All the source files will immediately go through linkage by mpicxx and nvcc compilers

* There will be **1 executable** called *ProximityCriteria* on your machine

* Please make sure that you type the command `make run`, this command will run the code single hosted, to adjust the number of processes please type `make run NUM_PROCESSES={number}`

* Before the actual run you can type the command `make runTest`, this will make the code to run sequential based on *input.txt* file, the output will be presented in file called *testOutput.txt*

* To clean the workspace enter the command: `make clean`

### How to Run The Code On Several Hosts: ###
* The process to run the code is the same as the single host appraoch but now, you will have to make sure that there are two updated executables on both machines, you can compile them both using the command `make`

* All the source files will immediately go through linkage by mpicxx and nvcc compilers

* There will be  **2 executable** called *ProximityCriteria* on seperated machines

* Please make sure you retrieve the IP address of both machines by typing `hostanme -I`, also it is important to put the master IP address on top of the file called *mf* from the executing machine.

* Please make sure that you type the command `make runOn2`, this command will run the code as distributed host

* To clean the workspace enter the command: `make clean` on both machines

**Before running the distributed approach make sure you putting the same updated files after compilation, and the correct ip addresses and the main host ip address is written on top in file called mf** 
