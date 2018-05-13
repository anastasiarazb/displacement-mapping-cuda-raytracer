#pragma once

#include "../cuda_app.h"

__device__ float3 screenToWorldRand(curandState_t *rand, uint ind, int width, int height);

__device__ float  prime_intersection(const Light &S, const float3 &start, const float3 &dir, uint id);

__device__ float4 LambertShading(InnerPoint hit_point, const Light &light, uint tri_num,
                                                                            const Material *material);
__device__ void   visualize_lights(float3 pixel, Pixel *gpu_accum, uint thread_id,
                                   Light *gpu_lights, uint32_t lights_num);
