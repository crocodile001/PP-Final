#ifndef CAMERAH
#define CAMERAH

#include "ray.h"
#include "cuda_def.h"

__device__ vec3 random_in_unit_disk(curandState *devStates, const int id)
{
    vec3 p;
    do
	{
        p = 2.0*vec3((curand_uniform_double(&devStates[id]) - DELTA), (curand_uniform_double(&devStates[id]) - DELTA), 0) - vec3(1, 1, 0);
    }
	while(dot(p, p) >= 1.0);
    return p;
}

class camera
{
public:
	__host__ __device__ camera() {}
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1)  //vfov is top to bottom degrees
    {
        time0 = t0;
        time1 = t1;
        lens_radius = aperture / 2;

        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        origin = lookfrom;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
        horizontal = 2*half_width*focus_dist*u;
        vertical = 2*half_height*focus_dist*v;
    }
    __device__ ray get_ray(curandState *devStates, const int id, float s, float t)
    {
        vec3 rd = lens_radius*random_in_unit_disk(devStates, id);
        vec3 offset = u*rd.x() + v*rd.y();
        float time = time0 + (curand_uniform_double(&devStates[id]) - DELTA) * (time1 - time0);
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset, time);
    }
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float time0, time1;
    float lens_radius;
};

//__device__ camera cam;

#endif