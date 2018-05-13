#pragma once
#include <cuda_runtime_api.h>

#pragma pack(push, 1)
union Pixel{
    struct {
        float r; float g; float b; float alpha;
    };
    float4 vec4;
    float3 vec3;


    __host__ __device__ void set(float r, float g, float b, float alpha);
    __host__ __device__ void set(float  r, float  g, float  b);
    __host__ __device__ void set(const float3 &v);
    __host__ __device__ void set(float4 v);

    __host__ __device__ void setv3(const float *v);
    __host__ __device__ void setv4(const float *v);
    __host__ __device__ void setPlusBackgrv4(const float *v, const float *back);
    __host__ __device__ void setPlusBackgrv3(const float *v, float alpha, const float *back);

    __host__ __device__ void safeAddColorConstAlpha(float3 color, float alpha);
    __host__ __device__ void safeAddColorConstAlpha(float4 color, float alpha);

    __host__ __device__ void safeAddColor(float3 color);
    __host__ __device__ void safeAddColor(float4 color, float scale);
};
#pragma pack(pop)
