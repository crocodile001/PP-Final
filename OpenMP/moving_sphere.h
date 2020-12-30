#ifndef MOVINGSPHEREH
#define MOVINGSPHEREH

#include "hitable.h"

class moving_sphere : public hitable
{
public:
    moving_sphere() {}
    moving_sphere(vec3 cen0, vec3 cen1, double t0, double t1, double r, material *m) : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {};
    virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
    vec3 center(double time) const;
    vec3 center0, center1;
    double time0, time1;
    double radius;
    material *mat_ptr;
};

vec3 moving_sphere::center(double time) const
{
    return center0 + ((time1 - time0)/(time1 - time0))*(center1 - center0);
}

bool moving_sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    vec3 oc = r.origin() - center(r.time());
    double a = dot(r.direction(), r.direction());
    double b = dot(oc, r.direction());
    double c = dot(oc, oc) - radius*radius;
    double discriminant = b*b - a*c;
    if(discriminant > 0)
    {
        double temp = (-b - sqrt(discriminant))/a;
        if(temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant))/a;
        if(temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif