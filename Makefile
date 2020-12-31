default: serial openmp mpi opencl

serial: serial.cpp
	g++ $< -o $@ -O3
openmp: openmp.cpp
	g++ $< -fopenmp -o $@ -O3
mpi: mpi.cpp
	mpic++ $< -o $@ -O3
opencl: opencl.cpp
	g++ $< -O3 -lOpenCL -m64 -w -o opencl
clean:
	rm -f *.ppm serial openmp mpi opencl
