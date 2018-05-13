#include "figures.h"
#include "../cuda_numerics/float3.h"

//----------------------------------------------------------------

void Triangle::setVertexAttrib_0(float3 p0, float3 n0, float2 uv0)
{
    this->p0  = p0;
    this->n0  = n0;
    this->uv0 = uv0;
}

void Triangle::setVertexAttrib_1(float3 p1, float3 n1, float2 uv1)
{
    this->p1  = p1;
    this->n1  = n1;
    this->uv1 = uv1;
}

void Triangle::setVertexAttrib_2(float3 p2, float3 n2, float2 uv2)
{
    this->p2  = p2;
    this->n2  = n2;
    this->uv2 = uv2;
}
