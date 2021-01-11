#include "library.h"
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(status, cmd)                                                     \
  {                                                                            \
    if (status != CL_SUCCESS) {                                                \
      printf("%s failed (%d)\n", cmd, status);                                 \
      exit(-1);                                                                \
    }                                                                          \
  }

using namespace std;
char *readSource(char *kernelPath) {
  cl_int status;
  FILE *fp;
  char *source;
  long int size;

  printf("Program file is: %s\n", kernelPath);

  fp = fopen(kernelPath, "rb");
  if (!fp) {
    printf("Could not open kernel file\n");
    exit(-1);
  }
  status = fseek(fp, 0, SEEK_END);
  if (status != 0) {
    printf("Error seeking to end of file\n");
    exit(-1);
  }
  size = ftell(fp);
  if (size < 0) {
    printf("Error getting file position\n");
    exit(-1);
  }

  rewind(fp);

  source = (char *)malloc(size + 1);

  int i;
  for (i = 0; i < size + 1; i++) {
    source[i] = '\0';
  }

  if (source == NULL) {
    printf("Error allocating space for the kernel source\n");
    exit(-1);
  }

  fread(source, 1, size, fp);
  source[size] = '\0';

  return source;
}

void initCL(cl_device_id *device, cl_context *context, cl_program *program) {
  // Set up the OpenCL environment
  cl_int status;

  // Discovery platform
  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform, NULL);
  CHECK(status, "clGetPlatformIDs");

  // Discover device
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
  CHECK(status, "clGetDeviceIDs");

  // Create context
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties)(platform), 0};
  *context = clCreateContext(props, 1, device, NULL, NULL, &status);
  CHECK(status, "clCreateContext");

  const char *source = readSource("opencl.cl");

  // Create a program object with source and build it
  *program = clCreateProgramWithSource(*context, 1, &source, NULL, NULL);
  CHECK(status, "clCreateProgramWithSource");
  status = clBuildProgram(*program, 1, device, NULL, NULL, NULL);
  CHECK(status, "clBuildProgram");

  return;
}

vec3 color(const ray &r, hitable *world, int depth) {
  hit_record rec;
  if (world->hit(r, 0.001, INT_MAX, rec)) {
    ray scattered;
    vec3 attenuation; //衰減
    if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return attenuation * color(scattered, world, depth + 1);
    } else {
      return vec3(0.0, 0.0, 0.0);
    }
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    double t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
  }
}

hitable *random_scene() {
  int n = 50000;
  hitable **list = new hitable *[n + 1];
  list[0] =
      new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
  int i = 1;
  for (int a = -10; a < 10; a++) {
    for (int b = -10; b < 10; b++) {
      float choose_mat = (double)rand() / (RAND_MAX + 1.0);
      vec3 center(a + 0.9 * (double)rand() / (RAND_MAX + 1.0), 0.2,
                  b + 0.9 * (double)rand() / (RAND_MAX + 1.0));
      if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
        if (choose_mat < 0.8) { // diffuse
          list[i++] = new moving_sphere(
              center,
              center + vec3(0, 0.5 * (double)rand() / (RAND_MAX + 1.0), 0), 0.0,
              1.0, 0.2,
              new lambertian(vec3((double)rand() / (RAND_MAX + 1.0) *
                                      (double)rand() / (RAND_MAX + 1.0),
                                  (double)rand() / (RAND_MAX + 1.0) *
                                      (double)rand() / (RAND_MAX + 1.0),
                                  (double)rand() / (RAND_MAX + 1.0) *
                                      (double)rand() / (RAND_MAX + 1.0))));
        } else if (choose_mat < 0.95) { // metal
          list[i++] = new sphere(
              center, 0.2,
              new metal(vec3(0.5 * (1 + (double)rand() / (RAND_MAX + 1.0)),
                             0.5 * (1 + (double)rand() / (RAND_MAX + 1.0)),
                             0.5 * (1 + (double)rand() / (RAND_MAX + 1.0))),
                        0.5 * (double)rand() / (RAND_MAX + 1.0)));
        } else { // glass
          list[i++] = new sphere(center, 0.2, new dielectric(1.5));
        }
      }
    }
  }

  list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
  list[i++] =
      new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
  list[i++] =
      new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

  return new hitable_list(list, i);
}

void hostFE(int nx, int ny, int ns, int *output) {

  size_t output_size = sizeof(int) * nx * ny * 3;
  output = (int *)malloc(output_size);

  cl_int status;
  cl_program program;
  cl_device_id device;
  cl_context context;
  initCL(&device, &context, &program);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
  cl_mem d_output =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "run", &status);
  CHECK(status, "clCreateKernel");

  clSetKernelArg(kernel, 0, sizeof(int), (void *)&nx);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&ny);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&ns);
  clSetKernelArg(kernel, 3, sizeof(int *), (void *)&d_output);

  size_t localws[2] = {8, 8};
  size_t globalws[2] = {nx, ny};

  status = clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalws, localws, 0,
                                  NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  clFinish(queue);
  clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size, (void *)output,
                      0, NULL, NULL);

  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
}

int main() {
  srand(time(NULL));

  int nx = 120;
  int ny = 80;
  int ns = 1;
  hitable *world = random_scene();
  fstream file;
  file.open("Hello.ppm", ios::out);
  file << "P3\n" << nx << " " << ny << "\n255\n";

  vec3 lookfrom(13, 2, 3);
  vec3 lookat(0, 0, 0);
  double dist_to_focus = 10.0;
  double aperture = 0.0; //光圈

  camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, double(nx) / double(ny),
             aperture, dist_to_focus, 0.0, 1.0);

  int *output;
  hostFE(nx, ny, ns, output);

  /*for(int j = ny-1; j >= 0; j -= 1)
      for(int i = 0; i < nx; i += 1)
      {
          vec3 col(0, 0, 0);
          for(int k = 0; k < ns; k += 1)
          {
              double u = double(i + (double)rand()/(RAND_MAX + 1.0)) /
  double(nx); double v = double(j + (double)rand()/(RAND_MAX + 1.0)) /
  double(ny); ray r = cam.get_ray(u, v); col += color(r, world, 0);
          }


          col /= double(ns);
          col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
          int ir = int(255.99 * col[0]);
          int ig = int(255.99 * col[1]);
          int ib = int(255.99 * col[2]);
          file << ir << " " << ig << " " << ib <<"\n";
      }
  }*/
}
