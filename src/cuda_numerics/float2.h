#pragma once

#include "cuda_numerics.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

#define FUNC_PREFIX __host__ __device__

static FUNC_PREFIX
float2 & operator += (float2 & a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

static FUNC_PREFIX
float2 & operator -= (float2 & a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

static FUNC_PREFIX
float2 & operator *= (float2 & a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

static FUNC_PREFIX
float2 & operator *= (float2 & a, float c)
{
    a.x *= c;
    a.y *= c;
    return a;
}

static FUNC_PREFIX
float2 & operator /= (float2 & a, float2 & b)
{
    a.x /= b.x;
    a.y /= b.y;
    return a;
}

static FUNC_PREFIX
float2 & operator /= (float2 & a, float c)
{
    a.x /= c;
    a.y /= c;
    return a;
}

static FUNC_PREFIX
float2 operator - (float2 a)
{
    return make_float2(-a.x, -a.y);
}

static FUNC_PREFIX
float2 operator + (float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

static FUNC_PREFIX
float2 operator - (float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

static FUNC_PREFIX
float2 operator * (float c, float2 a)
{
    return make_float2(a.x * c, a.y * c);
}

