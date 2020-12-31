default: serial openmp

serial: serial.cpp
	g++ $< -o $@
openmp: openmp.cpp
	g++ $< -fopenmp -o $@
clean:
	rm -f *.ppm serial openmp