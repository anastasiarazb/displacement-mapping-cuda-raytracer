#include "figures.h"
#include "cuda_runtime_api.h"
#include "../cuda_numerics/float3.h"
#include "../cuda_numerics/float2.h"
#include "stdio.h"

FUNC_PREF
void Baricentric::set(float u, float v)
{
    _beta  = u;
    _gamma = v;
    _alpha = 1.0f - u - v;
}

FUNC_PREF void  Baricentric::set(float alpha, float beta, float gamma)
{
    _alpha = alpha;
    _beta  = beta;
    _gamma = gamma;
}

FUNC_PREF RTIntersection::RTIntersection()
{
    t = INFINITY;
    success = false;
}

FUNC_PREF void RTIntersection::set(bool success, float t)
{
    this->success = success;
    this->t = t;
}

FUNC_PREF
void InnerPoint::displace(float displace)
{
    p += displace * norma(n);
}

FUNC_PREF
void InnerPoint::setInterpolate(InnerPoint A, InnerPoint B, float alpha)
{
    float complement = 1.0f - alpha;
    p  = complement * A.p  + alpha * B.p;
    n  = complement * A.n  + alpha * B.n;
    uv = complement * A.uv + alpha * B.uv;
}
