#include <stdio.h>
#include <pthread.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include "./library/sphere.h"
#include "./library/hitable_list.h"
#include "./library/camera.h"
#include "./library/material.h"
#include "./library/moving_sphere.h"

using namespace std;

vec3 color(const ray& r, hitable *world, int depth)
{
    hit_record rec;
    if(world->hit(r, 0.001, INT_MAX, rec))
    {
        ray scattered;
        vec3 attenuation;   //衰減
        if(depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
        {
            return attenuation*color(scattered, world, depth+1);
        }
        else
        {
            return vec3(0.0, 0.0, 0.0);
        }
    }
    else
    {
        vec3 unit_direction = unit_vector(r.direction());
        double t = 0.5*(unit_direction.y() + 1.0);
        return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

hitable *random_scene() {
    int n = 50000;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -10; a < 10; a++) {
        for (int b = -10; b < 10; b++) {
            float choose_mat = (double)rand()/(RAND_MAX + 1.0);
            vec3 center(a+0.9*(double)rand()/(RAND_MAX + 1.0),0.2,b+0.9*(double)rand()/(RAND_MAX + 1.0));
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new moving_sphere(
                        center, center+vec3(0, 0.5*(double)rand()/(RAND_MAX + 1.0), 0), 0.0, 1.0, 0.2,
                        new lambertian(vec3((double)rand()/(RAND_MAX + 1.0)*(double)rand()/(RAND_MAX + 1.0),
                                            (double)rand()/(RAND_MAX + 1.0)*(double)rand()/(RAND_MAX + 1.0),
                                            (double)rand()/(RAND_MAX + 1.0)*(double)rand()/(RAND_MAX + 1.0)))
                    );
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(
                        center, 0.2,
                        new metal(vec3(0.5*(1 + (double)rand()/(RAND_MAX + 1.0)),
                                       0.5*(1 + (double)rand()/(RAND_MAX + 1.0)),
                                       0.5*(1 + (double)rand()/(RAND_MAX + 1.0))),
                                  0.5*(double)rand()/(RAND_MAX + 1.0))
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

    return new hitable_list(list,i);
}

/*void* Thread_hits(void* num){

	long threadID = (long)num;
	unsigned long long number_in_circle = 0;
	double distance_squared, x, y;
	unsigned int seed = time(NULL) + threadID;
	unsigned long long tosses;

	tosses = (threadID == (thread_count-1)) ? final_number_of_tosses : number_of_tosses;

	for (unsigned long long toss = 0; toss < tosses; toss++){
		x = (double) xorshift(&seed) / RAND_MAX;
		y = (double) xorshift(&seed) / RAND_MAX;
		distance_squared = (x * x + y * y);
		if (distance_squared <= 1.0)
		    number_in_circle++;
	}

	pthread_mutex_lock(&mutex);
	hits += number_in_circle;
	pthread_mutex_unlock(&mutex);

	return NULL;
}*/

#define width 480
#define height 320
#define iter 10

int nx = width;
int ny = height;
int ns = iter;
unsigned char img[width][height][3];

int averow, extra;
long thread_count;

hitable *world;
vec3 lookfrom(13, 2, 3);
vec3 lookat(0, 0, 0);
double dist_to_focus = 10.0;
double aperture = 0.0;  //光圈
camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, double(nx)/double(ny), aperture, dist_to_focus, 0.0, 1.0);
//camera *cam;

void* run(void* num){

	long ID = (long)num;
	int start = (int)ID * averow;
    int end = start + averow;
	if(ID == thread_count - 1)
		end += extra;

	for(int i = start; i < end; i += 1)
        for(int j = 0; j < ny; j += 1)
        {
            vec3 col(0, 0, 0);
            for(int k = 0; k < ns; k += 1)
            {
                double u = double(i + (double)rand()/(RAND_MAX + 1.0)) / double(nx);
                double v = double(j + (double)rand()/(RAND_MAX + 1.0)) / double(ny);
                ray r = cam.get_ray(u, v);
                col += color(r, world, 0);
            }

            col /= double(ns);
            col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
            int ir = int(255.99 * col[0]);
            int ig = int(255.99 * col[1]);
            int ib = int(255.99 * col[2]);
            //file << ir << " " << ig << " " << ib <<"\n";
            img[i][j][0] = char(ir);
            img[i][j][1] = char(ig);
            img[i][j][2] = char(ib);
        }
	
	return NULL;

}

int main(int argc, char* argv[]){

	srand(9);

    //int nx = 120;
    //int ny = 80;
    //int ns = 1;
    world = random_scene();
    //fstream file;
    //file.open("Hello.ppm", ios::out);
    //file << "P3\n" << nx << " " << ny << "\n255\n";

    //vec3 lookfrom(13, 2, 3);
    //vec3 lookat(0, 0, 0);
    //double dist_to_focus = 10.0;
    //double aperture = 0.0;  //光圈

    //*cam = camera(lookfrom, lookat, vec3(0, 1, 0), 20, double(nx)/double(ny), aperture, dist_to_focus, 0.0, 1.0);

	long thread;
	pthread_t* thread_handles;

	thread_count = strtol(argv[1], NULL, 10);
	thread_handles = (pthread_t*) malloc(thread_count*sizeof(pthread_t));

	averow = nx / (int)thread_count;
    extra = nx % (int)thread_count;

	for(thread = 0; thread < thread_count; thread++)
		pthread_create(&thread_handles[thread], NULL, run, (void*)thread);

	for(thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);

	free(thread_handles);

	fstream file;
	file.open("Hello.ppm", ios::out);
	file << "P3\n" << nx << " " << ny << "\n255\n";

	for(int j = ny-1; j >= 0; j -= 1)
		for(int i = 0; i < nx; i += 1)
			file  << int(img[i][j][0]) << " " << int(img[i][j][1]) << " " << int(img[i][j][2]) << "\n";

	file.close();

	return 0;
}
