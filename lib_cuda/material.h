#ifndef MATERIALH
#define MATERIALH

#include "hitable.h"
#include "ray.h"
#include "cuda_def.h"

struct hit_record;

__device__ vec3 random_in_unit_sphere(curandState *devStates, const int id)
{
    vec3 p;
    do{
        p = 2.0*vec3((curand_uniform_double(&devStates[id]) - DELTA), (curand_uniform_double(&devStates[id]) - DELTA), (curand_uniform_double(&devStates[id]) - DELTA)) - vec3(1, 1, 1);
    }while(p.squared_length() >= 1.0);
    return p;
}

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2*dot(v, n)*n;
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if(discriminant > 0)
    {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

class material
{
public:
    __device__ virtual bool scatter(curandState *devStates, const int id, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
};

class dielectric : public material
{
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(curandState *devStates, const int id, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
		scattered = ray(vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
		return true;
        if(dot(r_in.direction(), rec.normal) > 0)
        {
            outward_normal = -1.0 * rec.normal;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / ref_idx;
            cosine = -1.0 * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        
        if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        {
            reflect_prob = schlick(cosine, ref_idx);
        }
        else
        {
            scattered = ray(rec.p, reflected, r_in.time());
            reflect_prob = 1.0;
        }
		if((curand_uniform_double(&devStates[id]) - DELTA) < reflect_prob)
        {
            scattered = ray(rec.p, reflected, r_in.time());
        }
        else
        {
            scattered = ray(rec.p, refracted, r_in.time());
        }
        return true;
    }
    float ref_idx;
};

class lambertian : public material
{
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(curandState *devStates, const int id, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const
    {
		attenuation = vec3(1.0, 1.0, 1.0);
		scattered = ray(vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0);
        vec3 light(0, 2, 0);
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(devStates, id);
		// very weird floating point problem
		vec3 tmp(-1.0f*rec.p.x(), -1.0f*rec.p.y(), -1.0f*rec.p.y());
		light += tmp;
		light /= light.squared_length();
		////////////////////////////////		
        scattered = ray(rec.p, rec.normal + light + random_in_unit_sphere(devStates, id), r_in.time());
		attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public material
{
public:
    __device__ metal(const vec3& a, float f) : albedo(a) {if(f < 1) fuzz = f; else fuzz = 1;}
    __device__ virtual bool scatter(curandState *devStates, const int id, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const
    {
		attenuation = vec3(1.0, 1.0, 1.0);
		scattered = ray(vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0);
        vec3 light(0, 2, 0);
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		// very weird floating point problem
		vec3 tmp(-1.0f*rec.p.x(), -1.0f*rec.p.y(), -1.0f*rec.p.y());
		light += tmp;
		light /= light.squared_length();
		////////////////////////////////
        scattered = ray(rec.p, reflected + light + fuzz*random_in_unit_sphere(devStates, id), r_in.time());
		attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    vec3 albedo;
    float fuzz;
};

#endif