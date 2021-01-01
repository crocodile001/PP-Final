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
#include <mpi.h>

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

int main(int argc, char **argv)
{
    int rank, size, dx, averow, extra, start, end;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //printf("rank = %d, size = %d\n", rank, size);

    srand(time(NULL));

    int nx = 480;
    int ny = 320;
    int ns = 100;
    hitable *world = random_scene();

    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    double dist_to_focus = 10.0;
    double aperture = 0.0;  //光圈
    int img[nx][ny][3];
    

    camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, double(nx)/double(ny), aperture, dist_to_focus, 0.0, 1.0);
    
    averow = nx / size;
    extra = nx % size;
    if(rank == size - 1)
        dx = averow + extra;
        //dx = (rank <= extra)? averow + 1 : averow;
    else
        dx = averow;
    
    start = rank * averow;
    end = start + dx;
    //printf("%d %d\n", rank, dx);
    //printf("%d %d %d\n", rank, start, end);

    MPI_Request *requests;
    MPI_Status *status2;
    if(rank == 0){

        requests = (MPI_Request*)malloc(sizeof(MPI_Request) * (size-1));
        status2 = (MPI_Status*)malloc(sizeof(MPI_Status) * (size-1));

        int rows;
        int offset = averow;
        for (int source = 1; source < size; source++){

            //rows = (source <= extra)? averow + 1 : averow;
            rows = (source == size - 1)? averow + extra : averow;
            MPI_Irecv(&img[offset][0][0], rows * ny * 3, MPI_INT, source, 1, MPI_COMM_WORLD, &requests[source-1]);
            offset = offset + rows;

        }

    }

    for(int j = 0; j < ny; j += 1)
        for(int i = start; i < end; i += 1)
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
            img[i][j][0] = ir;
            img[i][j][1] = ig;
            img[i][j][2] = ib;
        }

    if(rank == 0){

        fstream file;
        file.open("Hello.ppm", ios::out);
        file << "P3\n" << nx << " " << ny << "\n255\n";
        MPI_Waitall(size-1, requests, status2);

        for(int j = ny-1; j >= 0; j -= 1)
            for(int i = 0; i < nx; i += 1)
                file  << img[i][j][0] << " " << img[i][j][1] << " " << img[i][j][2] << "\n";

        file.close();

    }
    else{

        MPI_Request req;
        MPI_Isend(&img[start][0][0], dx * ny * 3, MPI_INT, 0, 1, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, &status);
    
    }

    MPI_Finalize();
    return 0;
}
