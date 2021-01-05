# PP Final
Monte Carlo Ray Tracer

**Compile :**

```
$ make
```

**Serial :**

```
$ ./serial
```

**Pthread :**

```
$ ./pthread <numOfthread>
```

**Openmp :**

```
$ ./openmp
```

**MPI :**

```
$ mpic++ mpi.cpp -o mpi_<method>
$ mpiexec -n 4 ./mpi_<method>
```

or

```
$ mpicxx mpi.cpp -o mpi_<method>
$ mpirun -np 4 --hostfile host/hosts ./mpi_<method>
```

if hosts file is not work, please rewrite it

**Cuda :**

```
$ ./cuda
```

**Opencl :**

- working

