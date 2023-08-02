# Parallel_Computation - Final Project

The code is dealing with several Parallel Computation methodologies, *CUDA* *OpenMp* threads and *MPI* processes.

* *The single host approach* -> The host will run two processes, one of the prcesses is reading and preparing the data  by the help of the OMP threads and afterwards send it to the other process.
Each process calculates the histogram of the calues within the large data set by operating CUDA threads on the GPU.
When each process is finished to calculte the histogram it, the slave process will send the sum to the master and then the the master will sum up the two histograms by the help of OMP threads.

* *The distributed host approach* -> The program will be excuted the same as the single host approach, but on different hosts on the same LAN network.

### How to Run The Code On Single Host: ###
* There is a Makefile supplied to you, all you need to do is to open the terminal locally under the folder's path and then enter the command: `make`

* All the source files will immediately go through linkage by mpicxx and nvcc compilers

* There will be **1 executable** called *mpiCudaOpemMP* on your machine

* Please make sure that you type the command `make run`, this command will run the code single hosted

* To clean the workspace enter the command: `make clean`

### How to Run The Code On Several Hosts: ###
* The process to run the code is the same as the single host appraoch but now, you will have to make sure that there are two updated executables on both machines, you can compile them both using the command `make`

* All the source files will immediately go through linkage by mpicxx and nvcc compilers

* There will be  **2 executable** called *mpiCudaOpemMP* on seperated machines

* Please make sure that you type the command `make runOn2`, this command will run the code as distributed host

* To clean the workspace enter the command: `make clean` on both machines

**Before running the distributed approach make sure you putting the same updated files after compilation, and the correct ip addresses and the main host ip address is written on top in file called mf** 