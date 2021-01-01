#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "cuda_def.h"

using namespace std;

class vec3
{
public:
    __device__ vec3(){}
    __device__ vec3(double e0, double e1, double e2) {e[0] = e0; e[1] = e1; e[2] = e2;}
    __device__ inline double x() const {return e[0];}
    __device__ inline double y() const {return e[1];}
    __device__ inline double z() const {return e[2];}
    __device__ inline double r() const {return e[0];}
    __device__ inline double g() const {return e[1];}
    __device__ inline double b() const {return e[2];}

    __device__ inline const vec3& operator+() const {return *this;}
    __device__ inline vec3 operator-() const {return vec3(-e[0], -e[1], -e[2]);}
    __device__ inline double operator[](int i) const {return e[i];}
    __device__ inline double& operator[](int i){{return e[i];}}

    __device__ inline vec3& operator+=(const vec3 &v2);
    __device__ inline vec3& operator-=(const vec3 &v2);
    __device__ inline vec3& operator*=(const vec3 &v2);
    __device__ inline vec3& operator/=(const vec3 &v2);
    __device__ inline vec3& operator*=(const double t);
    __device__ inline vec3& operator/=(const double t);

    __device__ inline double length() const
    {
        return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    }
    __device__ inline double squared_length() const
    {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + DELTA;
    }
    __device__ inline void make_unit_vector();

    double e[3];
};

__device__ inline void vec3::make_unit_vector()
{
    double k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] +e[2]*e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__device__ inline vec3 operator+(const vec3 &v1,const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__device__ inline vec3 operator-(const vec3 &v1,const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__device__ inline vec3 operator*(const vec3 &v1,const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__device__ inline vec3 operator/(const vec3 &v1,const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__device__ inline vec3 operator*(double t, const vec3 &v)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ inline vec3 operator/(const vec3 &v, double t)
{
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__device__ inline vec3 operator*(const vec3 &v, double t)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ inline double dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2]; 
}

__device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                 (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

__device__ inline vec3& vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__device__ inline vec3& vec3::operator-=(const vec3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__device__ inline vec3& vec3::operator*=(const vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__device__ inline vec3& vec3::operator/=(const vec3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__device__ inline vec3& vec3::operator*=(const double t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__device__ inline vec3& vec3::operator/=(const double t)
{
    double k = 1.0/t;
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}
