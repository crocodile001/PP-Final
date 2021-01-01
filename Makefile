default: serial openmp mpi_nonblock mpi_gather opencl cuda

serial: serial.cpp
	g++ $< -o $@ -O3
openmp: openmp.cpp
	g++ $< -fopenmp -o $@ -O3
mpi_nonblock: mpi_nonblock.cpp
	mpicxx $< -o $@ -O3
mpi_gather: mpi_gather.cpp
	mpicxx $< -o $@ -O3
opencl: opencl.cpp
	g++ $< -O3 -lOpenCL -m64 -w -o opencl
cuda: cuda.cu
	nvcc -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 $< -o $@
clean:
	rm -f *.ppm serial openmp mpi_nonblock mpi_gather opencl cuda
