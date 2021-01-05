default: serial pthread openmp mpi_nonblock mpi_gather cuda opencl

serial: serial.cpp
	g++ $< -o $@ -O3
pthread: pthread.cpp
	g++ -pthread -O3 $< -o $@
openmp: openmp.cpp
	g++ $< -fopenmp -o $@ -O3
mpi_nonblock: mpi_nonblock.cpp
	mpicxx $< -o $@ -O3
mpi_gather: mpi_gather.cpp
	mpicxx $< -o $@ -O3
cuda: cuda.cu
	nvcc -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 $< -o $@
opencl: opencl.cpp
	g++ $< -O3 -lOpenCL -m64 -w -o opencl
clean:
	rm -f *.ppm serial pthread openmp mpi_nonblock mpi_gather cuda opencl
