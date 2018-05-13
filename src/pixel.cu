#include "pixel.h"
#include "math.h"

#define UGLY_CAST(T) *(T *)&

__host__ __device__ void Pixel::set(float r, float g, float b, float alpha) {
    this->r = r;
    this->g = g;
    this->b = b;
    this->alpha = alpha;
}

__host__ __device__ void Pixel::set(float  r, float  g, float  b) {
    this->r = r;
    this->g = g;
    this->b = b;
    alpha = 1.0f;
}

__host__ __device__ void Pixel::set(const float3 &v) {
    vec3 = v;
}

__host__ __device__ void Pixel::set(float4 v) {
    vec4 = v;
}

__host__ __device__ void Pixel::setv3(const float *v) {
    set(v[0], v[1], v[2]);
}

__host__ __device__ void Pixel::setv4(const float *v) {
    set(v[0], v[1], v[2], v[3]);
}

__host__ __device__ void Pixel::setPlusBackgrv4(const float *v, const float *back) {
    setPlusBackgrv3(v, v[3], back);
}


__host__ __device__ void Pixel::setPlusBackgrv3(const float *v, float alpha, const float *back)
{
    this->r = back[0]+(v[0] - back[0])*alpha;
    this->g = back[1]+(v[1] - back[1])*alpha;
    this->b = back[2]+(v[2] - back[2])*alpha;
    this->alpha = alpha;
}

//----------------------------------------------------------------------------------------------------

__host__ __device__ void Pixel::safeAddColorConstAlpha(float3 color, float alpha)
{
    r = fmin(1.0f, r + (color.x - r)*alpha);
    g = fmin(1.0f, g + (color.y - g)*alpha);
    b = fmin(1.0f, b + (color.z - b)*alpha);
}

__host__ __device__ void Pixel::safeAddColorConstAlpha(float4 color, float alpha)
{
    r = fmin(1.0f, r + (color.x - r)*alpha);
    g = fmin(1.0f, g + (color.y - g)*alpha);
    b = fmin(1.0f, b + (color.z - b)*alpha);
}

__host__ __device__ void Pixel::safeAddColor(float3 color)
{
    r = fmin(1.0f, r + color.x);
    g = fmin(1.0f, g + color.y);
    b = fmin(1.0f, b + color.z);
}

__host__ __device__ void Pixel::safeAddColor(float4 color, float scale)
{
    r += scale * fmin(1.0f, color.x);
    g += scale * fmin(1.0f, color.y);
    b += scale * fmin(1.0f, color.z);
}

