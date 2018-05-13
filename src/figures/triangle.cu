#include "figures.h"
#include "cuda_runtime_api.h"
#include "../cuda_numerics/float3.h"
#include "../cuda_numerics/float2.h"
#include "stdio.h"

#define EPSILON 10e-15f

FUNC_PREF
Triangle make_triangle(float3 p0, float3 p1, float3 p2)
{
    Triangle T;
    T.p0 = p0;
    T.p1 = p1;
    T.p2 = p2;
    return T;
}

FUNC_PREF
void Triangle::set(float3 p0, float3 p1, float3 p2)
{
    this->p0 = p0;
    this->p1 = p1;
    this->p2 = p2;
}

FUNC_PREF void Triangle::set(InnerPoint A, InnerPoint B, InnerPoint C)
{
    p0  = A.p;
    n0  = A.n;
    uv0 = A.uv;

    p1  = B.p;
    n1  = B.n;
    uv1 = B.uv;

    p2  = C.p;
    n2  = C.n;
    uv2 = C.uv;
}

/*   p0
     | \
     p1-p2 */

FUNC_PREF
float3 Triangle::normal() const //p0, p1, p2 обходятся по часовой стрелке, но ск правостороння, z смотрит от нас
{
    return norma(cross(p1-p0, p2-p0));
}

FUNC_PREF
Ray reflect(Ray ray, float3 normal)
{
    return make_ray(ray.org, ray.dir - 2 * dot(ray.dir, normal) * normal);
}

//-----------------------------------------------------------------
FUNC_PREF
float3 Triangle::interpolatePoint(Baricentric coords) const
{
    return coords.alpha() * p0 + coords.beta() * p1 + coords.gamma() * p2;
}

FUNC_PREF
float3 Triangle::interpolateNormal(Baricentric coords) const
{
    return coords.alpha() * n0 + coords.beta() * n1 + coords.gamma() * n2;
}

FUNC_PREF
float2 Triangle::interpolateUV(Baricentric coords) const
{
    return coords.alpha() * uv0 + coords.beta() * uv1 + coords.gamma() * uv2;
}

FUNC_PREF InnerPoint Triangle::interpolate(Baricentric coords) const
{
    InnerPoint P;
    P.p  = interpolatePoint (coords);
    P.n  = interpolateNormal(coords);
    P.uv = interpolateUV    (coords);
    return P;
}

FUNC_PREF InnerPoint Triangle::interpolate(float u, float v) const
{
    InnerPoint P;
    float alpha = 1.0f - u - v;
    P.p  = alpha *  p0 + u *  p1 + v *  p2;
    P.n  = alpha *  n0 + u *  n1 + v *  n2;
    P.uv = alpha * uv0 + u * uv1 + v * uv2;
    return P;
}

FUNC_PREF void Triangle::displace(float h0, float h1, float h2)
{
//    p0 = p0 + norma(n0)*h0;
//    p1 = p1 + norma(n1)*h1;
//    p2 = p2 + norma(n2)*h2;
    p0 = p0 + n0*h0;
    p1 = p1 + n1*h1;
    p2 = p2 + n2*h2;
}

FUNC_PREF void Triangle::setDefaultNormals()
{
    n0 = n1 = n2 = normal();
}

FUNC_PREF Triangle Triangle::getMicrotriangle(Baricentric uva, Baricentric uvb, Baricentric uvc) const
{
    Triangle T;
    T.p0 = interpolatePoint(uva);
    T.p1 = interpolatePoint(uvb);
    T.p2 = interpolatePoint(uvc);

    T.n0 = interpolateNormal(uva);
    T.n1 = interpolateNormal(uvb);
    T.n2 = interpolateNormal(uvc);

    T.uv0 = interpolateUV(uva);
    T.uv1 = interpolateUV(uvb);
    T.uv2 = interpolateUV(uvc);

    return T;
}

FUNC_PREF InnerPoint Triangle::getVertex0() const
{
    InnerPoint P;
    P.p  =  p0;
    P.n  =  n0;
    P.uv = uv0;
    return P;
}
FUNC_PREF InnerPoint Triangle::getVertex1() const
{
    InnerPoint P;
    P.p  =  p1;
    P.n  =  n1;
    P.uv = uv1;
    return P;
}
FUNC_PREF InnerPoint Triangle::getVertex2() const
{
    InnerPoint P;
    P.p  =  p2;
    P.n  =  n2;
    P.uv = uv2;
    return P;
}

FUNC_PREF void Triangle::print() const
{
    printf("--\nTriangle:"
           "Points:         <{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}>\n"
           "Normals:        <{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}>\n"
           "Texture coords: <{%f, %f}, {%f, %f}, {%f, %f}>\n",
           p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z,
           n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z,
           uv0.x, uv0.y,     uv1.x, uv1.y,    uv2.x, uv2.y
           );
}
