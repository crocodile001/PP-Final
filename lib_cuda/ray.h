#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3& a, const vec3& b, double t)
    {
        A = a;
        B = b;
        _time = t;
    }
    __device__ vec3 origin() const { return A; }
    __device__ vec3 direction() const { return B; }
    __device__ double time() const { return _time; }
    __device__ vec3 point_at_parameter(double t) const { return A + t*B; }

    vec3 A;
    vec3 B;
    double _time;
};

#endif