#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include "./lib_cuda/sphere.h"
#include "./lib_cuda/hitable_list.h"
#include "./lib_cuda/camera.h"
#include "./lib_cuda/material.h"
#include "./lib_cuda/moving_sphere.h"
#include <cuda.h>
#include <curand_kernel.h>
#include "./lib_cuda/cuda_def.h"
#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4
#define DEPTH 30
#define MIRROR 50

using namespace std;

__device__ vec3 color(curandState *devStates, int id, const ray& r, int depth)
{
    hit_record rec;
	vec3 attenuation;   //衰減
	ray scattered;
	
	if(world->hit(r, 0.001, MIRROR, rec))
    {
        if(depth < DEPTH && rec.mat_ptr->scatter(devStates, id, r, rec, attenuation, scattered))
        {
            return attenuation*color(devStates, id, scattered, depth+1);
        }
        else
        {
            return vec3(0.0, 0.0, 0.0);
        }
    }
    else
    {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
	
	/*
	int i;
	vec3 att(1.0, 1.0, 1.0);
	bool h = world->hit(r, 0.001, M, rec);
	bool s = true;
	for(i=0; i<DEPTH && (s && h); ++i)
	{
		s = rec.mat_ptr->scatter(devStates, id, r, rec, attenuation, scattered);
		att *= attenuation;
		h = world->hit(r, 0.001, M, rec);
	}
	if(h)
		return vec3(0.0, 0.0, 0.0);
	else
    {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
		vec3 ret = (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
        return att*ret;
    }
	*/
}

__global__ void random_scene_kernel()
{
	// seed the scene
	curandState sceneState;
	curand_init(54321, 0, 0, &sceneState);
    int n = 50000;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -10; a < 10; a++) {
        for (int b = -10; b < 10; b++) {
            float choose_mat = (curand_uniform_double(&sceneState) - DELTA);
            vec3 center(a+0.9*(curand_uniform_double(&sceneState) - DELTA), 0.2, b+0.9*(curand_uniform_double(&sceneState) - DELTA));
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new moving_sphere(
                        center, center+vec3(0, 0.5*(curand_uniform_double(&sceneState) - DELTA), 0), 0.0, 1.0, 0.2,
                        new lambertian(vec3((curand_uniform_double(&sceneState) - DELTA)*(curand_uniform_double(&sceneState) - DELTA),
                                            (curand_uniform_double(&sceneState) - DELTA)*(curand_uniform_double(&sceneState) - DELTA),
                                            (curand_uniform_double(&sceneState) - DELTA)*(curand_uniform_double(&sceneState) - DELTA)))
                    );
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(
                        center, 0.2,
                        new metal(vec3(0.5*(1 + (curand_uniform_double(&sceneState) - DELTA)),
                                       0.5*(1 + (curand_uniform_double(&sceneState) - DELTA)),
                                       0.5*(1 + (curand_uniform_double(&sceneState) - DELTA))),
                                  0.5*(curand_uniform_double(&sceneState) - DELTA))
                    );
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    world = new hitable_list(list,i);
}

__global__ void cam_kernel(camera *cam, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1)
{
	vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	*cam = camera(lookfrom, lookat, vup, 20, vfov, aspect, focus_dist, t0, t1);
}

__global__ void rand_kernel(curandState *devStates, int nx, int ny)
{
	// threads use the same seed but different sequence, no offset
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = y * nx + x;
    // __device__ void curand_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t *state)
	if(x < nx && y < ny)
		curand_init(0, id, x, &devStates[id]);
}

__global__ void pixel_kernel(camera *cam, curandState *devStates, int *img, int nx, int ny, int ns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = y * nx + x;
	vec3 col(0, 0, 0);
	float u, v;
	int k;
	if(x < nx && y < ny)
	{
		for(k = 0; k < ns; ++k)
		{		
			// __device__ unsigned int curand(curandState_t *state)
			// __device__ float curand_uniform_double(curandState_t *state)
			u = (float)(x + curand_uniform_double(&devStates[id]) - DELTA) / (float)nx;
			v = (float)(y + curand_uniform_double(&devStates[id]) - DELTA) / (float)ny;
			ray r = cam->get_ray(devStates, id, u, v);
			col += color(devStates, id, r, 0);
		}
		col /= (float)ns;
		col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
		img[id*3] = (int)(255.99 * col[0]);
		img[id*3 + 1] = (int)(255.99 * col[1]);
		img[id*3 + 2] = (int)(255.99 * col[2]);
	}
}

int main()
{
    int nx = 120;
    int ny = 80;
    int ns = 10;
    float dist_to_focus = 10.0;
    float aperture = 0.0;  //光圈
	fstream file;
    file.open("Hello.ppm", ios::out);
    file << "P3\n" << nx << " " << ny << "\n255\n";
	///////////////////////////////////////////////////////////
	random_scene_kernel<<<1, 1>>>();
	//cudaDeviceSynchronize();
	///////////////////////////////////////////////////////////
	camera *cam;
	cudaMalloc((void**)&cam, sizeof(camera));
    cam_kernel<<<1, 1>>>(cam, 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0);
    //cudaDeviceSynchronize();
	///////////////////////////////////////////////////////////	
	curandState *devStates;
	cudaMalloc((void **)&devStates, nx*ny*sizeof(curandState));
	int gx = nx/BLOCK_WIDTH, gy = ny/BLOCK_HEIGHT;
	if(nx - gx*BLOCK_WIDTH != 0) ++gx;
	if(ny - gy*BLOCK_HEIGHT != 0) ++gy;
	dim3 dimGrid(gx, gy);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	//printf("%d %d %d %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
	rand_kernel<<<dimGrid, dimBlock>>>(devStates, nx, ny);
	//cudaDeviceSynchronize();
	///////////////////////////////////////////////////////////
	int *img, *devImg;
	img = (int*)malloc(3*nx*ny*sizeof(int));
	cudaMalloc((void **)&devImg, 3*nx*ny*sizeof(int));
	cudaMemset(devImg, 0, 3*nx*ny*sizeof(int));
	pixel_kernel<<<dimGrid, dimBlock>>>(cam, devStates, devImg, nx, ny, ns);
	cudaDeviceSynchronize();
	cudaMemcpy(img, devImg, 3*nx*ny*sizeof(int), cudaMemcpyDeviceToHost);	
	for(int i=0; i<3*nx*ny; i+=3)
		file << img[i] << " " << img[i+1] << " " << img[i+2] << "\n";
	free(img);
	cudaFree(devImg);
	cudaFree(devStates);
}
