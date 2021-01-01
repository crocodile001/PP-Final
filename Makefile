default: serial openmp mpi_nonblock mpi_gather opencl

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
clean:
	rm -f *.ppm serial openmp mpi_nonblock mpi_gather opencl
