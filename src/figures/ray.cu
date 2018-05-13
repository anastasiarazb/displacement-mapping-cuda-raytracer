#include "figures.h"
#include "figures.h"
#include "cuda_runtime_api.h"
#include "../cuda_numerics/float3.h"
#include "stdio.h"

FUNC_PREF
float3 Ray::point(float t)
{
    return org + dir*t;
}

//----------------------------------Constructors-----------------------------------------

FUNC_PREF Ray::Ray()
{
    tmin = 0.0f;
    tmax = INFINITY;
}

FUNC_PREF Ray::Ray(float3 start, float3 dir)
{
    tmin = 0.0f;
    tmax = INFINITY;
    this->org = start;
    this->dir = dir;
}

FUNC_PREF Ray::Ray(float3 start, float3 dir, float tmin)
{
    this->org  = start;
    this->dir  = dir;
    this->tmin = tmin;
    this->tmax = INFINITY;
}

FUNC_PREF Ray::Ray(float3 start, float3 dir, float tmin, float tmax)
{
    this->org  = start;
    this->dir  = dir;
    this->tmin = tmin;
    this->tmax = tmax;
}

FUNC_PREF
Ray make_ray(float3 org, float3 dir)
{
    Ray new_ray;
    new_ray.org  = org;
    new_ray.dir  = dir;
    new_ray.tmin = 0;
    new_ray.tmax = INFINITY;
    return new_ray;
}

FUNC_PREF
Ray make_ray(float3 org, float3 dir, float tmin)
{
    Ray new_ray;
    new_ray.org  = org;
    new_ray.dir  = dir;
    new_ray.tmin = tmin;
    new_ray.tmax = INFINITY;
    return new_ray;
}

FUNC_PREF
Ray make_ray(float3 org, float3 dir, float tmin, float tmax)
{
    Ray new_ray;
    new_ray.org  = org;
    new_ray.dir  = dir;
    new_ray.tmin = tmin;
    new_ray.tmax = tmax;
    return new_ray;
}

FUNC_PREF
void Ray::set(float3 org, float3 dir, float tmin, float tmax)
{
    this->org  = org;
    this->dir  = dir;
    this->tmin = tmin;
    this->tmax = tmax;
}

FUNC_PREF
void Ray::set(float3 org, float3 dir, float tmin)
{
    this->org  = org;
    this->dir  = dir;
    this->tmin = tmin;
    this->tmax = INFINITY;
}

FUNC_PREF
void Ray::set(float3 org, float3 dir)
{
    this->org  = org;
    this->dir  = dir;
    this->tmin = 0;
    this->tmax = INFINITY;
}

FUNC_PREF void Ray::print() const
{
    printf("-- Ray: <org, dir> = <{%f, %f, %f}, {%f, %f, %f}>\n",
           org.x, org.y, org.z,
           dir.x, dir.y, dir.z);
}
