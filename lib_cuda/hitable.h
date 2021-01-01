#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hit_record
{
    double t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    bool is_light;
};

class hitable
{
public:
    __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

__device__ hitable *world;

#endif