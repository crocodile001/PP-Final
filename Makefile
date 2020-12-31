default: serial openmp mpi

serial: serial.cpp
	g++ $< -o $@ -O3
openmp: openmp.cpp
	g++ $< -fopenmp -o $@ -O3
mpi: mpi.cpp
	mpic++ $< -o $@ -O3
clean:
	rm -f *.ppm serial openmp mpi