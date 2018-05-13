#ifndef FIGURES_H
#define FIGURES_H
#include <iostream>
#include <cuda_runtime_api.h>
#include "../bounding_box/boundingbox.h"
#include "../textures/texture.h"

#define FUNC_PREF __host__ __device__

struct Ray {
    float3 org; //Стартовая точка луча
    float3 dir; //Направление луча
    // t - параметр в уравнении x = org + t*dir
    float tmin; //Минимальный  параметр t
    float tmax; //Максимальный параметр t

    FUNC_PREF float3   point(float t);
    FUNC_PREF Ray    reflect(Ray ray, float3 normal);

    FUNC_PREF Ray();
    FUNC_PREF Ray(float3 start, float3 dir);
    FUNC_PREF Ray(float3 start, float3 dir, float tmin);
    FUNC_PREF Ray(float3 start, float3 dir, float tmin, float tmax);

    FUNC_PREF
    void set(float3 start, float3 dir);
    FUNC_PREF
    void set(float3 org, float3 dir, float tmin);
    FUNC_PREF
    void set(float3 org, float3 dir, float tmin, float tmax);
    FUNC_PREF void print() const;

//    создать отраженный луч(org, dir, hit_t, bounceAttenuation);
};

bool __device__ rayAABBIntersection(float3 start, float3 dir, float3 aabbmin, float3 aabbmax);

FUNC_PREF
Ray make_ray(float3 origin, float3 dir);
FUNC_PREF
Ray make_ray(float3 origin, float3 dir, float tmin);
FUNC_PREF
Ray make_ray(float3 origin, float3 dir, float tmin, float tmax);
/*--------------------------------------------------------------------------------*/

struct Light
{
    float4 color;
    float3 center;
    float Radius;

    float intensity;
/*  TODO: сделать затухание по квадратичному закону Atten = 1/( att0 + att1 * d + att2 * d*d) \
    float 	mAttenuationConstant Constant light attenuation factor. att0\
    float 	mAttenuationLinear   Linear light attenuation factor. att1\
    float 	mAttenuationQuadratic att2\
    http://assimp.sourceforge.net/lib_html/structai_light.html */
};

void initLights(Light lights[]);
void initLights(Light *&lights, unsigned int &num_of_lights, BoundingBox bb);

//-------------------------------------------------

struct InnerPoint {
    float3 p;
    float3 n;
    float2 uv;

    FUNC_PREF
    void displace(float displace);
    FUNC_PREF void setInterpolate(InnerPoint A, InnerPoint B, float alpha);
};

class Baricentric {
    float _alpha, _beta, _gamma;
public:
    FUNC_PREF float alpha() const {return _alpha;}
    FUNC_PREF float beta()  const {return _beta;}
    FUNC_PREF float gamma() const {return _gamma;}
    FUNC_PREF void  set(float u, float v);
    FUNC_PREF void  set(float alpha, float beta, float gamma);
};

struct RTIntersection
{
    bool success;
    float t;
    InnerPoint intersectionPoint;
    Baricentric coords;

    FUNC_PREF RTIntersection();
    FUNC_PREF void set(bool success, float t);
};
/*--------------------------------------------------------------------------------*/
struct Triangle {
    float3 p0, p1, p2; //Координаты вершин: стандартный обход против часовой стрелки
    float3 n0, n1, n2; //Нормали
    float2 uv0, uv1, uv2; //Текстурные координаты

    FUNC_PREF bool intersect(const Ray &ray, float &t) const;

    FUNC_PREF void set(float3 p0, float3 p1, float3 p2);
    FUNC_PREF void set(InnerPoint A, InnerPoint B, InnerPoint C);

    FUNC_PREF
    float3 normal() const;

    void setVertexAttrib_0(float3 p0, float3 n0, float2 uv0);
    void setVertexAttrib_1(float3 p1, float3 n1, float2 uv1);
    void setVertexAttrib_2(float3 p2, float3 n2, float2 uv2);

    FUNC_PREF float3     interpolatePoint (Baricentric coords) const;
    FUNC_PREF float3     interpolateNormal(Baricentric coords) const;
    FUNC_PREF float2     interpolateUV    (Baricentric coords) const;
    FUNC_PREF InnerPoint interpolate      (Baricentric coords) const;
    FUNC_PREF InnerPoint interpolate      (float u, float v) const;
    FUNC_PREF void       displace         (float h0, float h1, float h2);
    FUNC_PREF void       setDefaultNormals();
    FUNC_PREF Triangle   getMicrotriangle(Baricentric uva, Baricentric uvb, Baricentric uvc) const;
    FUNC_PREF InnerPoint getVertex0() const;
    FUNC_PREF InnerPoint getVertex1() const;
    FUNC_PREF InnerPoint getVertex2() const;
    FUNC_PREF void print() const;
};

FUNC_PREF
RTIntersection rayTriangleIntersection(const Ray &ray, const Triangle &triangle);
FUNC_PREF
RTIntersection rayTriangleIntersectionPrimary(const Ray &ray, const Triangle &triangle);

__device__
RTIntersection rayDisplacedTriIntersection(const Ray &ray, const Triangle &triangle, \
                                           const Texture<float> &displaces, \
                                           int subdiv_param, float max_height, uint thread_idx);
__device__
RTIntersection rayDisplacedTriIntersection1(const Ray &ray, const Triangle &triangle, \
                                           const Texture<float> &displaces, \
                                           int subdiv_param);
/*--------------------------------------------------------------------------------*/
struct Sphere
{
    float4 color;
    float3 center;
    float Radius;

    float reflect;
    float ambient;
    float diffuse = 1.f;
};

void initSpheres(Sphere spheres[]);

#endif // FIGURES_H
