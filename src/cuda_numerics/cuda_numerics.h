#pragma once

#include <cmath>
#include <inttypes.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define FUNC_PREFIX __host__ __device__

template<typename T> static FUNC_PREFIX
T clamp(T x, T min, T max)
{
    if (x > max) x = max;
    if (x < min) x = min;
    return x;
}

template<typename T> static FUNC_PREFIX
T clampUp(T x, T max)
{
    if (x > max) x = max;
    return x;
}

template<typename T> static FUNC_PREFIX
T clampDown(float x, float min)
{
    if (x < min) x = min;
    return x;
}

static FUNC_PREFIX
double fmin(float a1, float a2, float a3)
{
    return fmin(fmin(a1, a2), a3);
}

static FUNC_PREFIX
double fmax(float a1, float a2, float a3)
{
    return fmax(fmax(a1, a2), a3);
}

static FUNC_PREFIX
double fmin(float a1, float a2, float a3, float a4, float a5, float a6)
{
    return fmin(fmin(a1, a2, a3), fmin(a4, a5, a6));
}

static FUNC_PREFIX
double fmax(float a1, float a2, float a3, float a4, float a5, float a6)
{
    return fmax(fmax(a1, a2, a3), fmax(a4, a5, a6));
}
