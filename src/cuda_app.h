#pragma once

#include "cuda_runtime_api.h"
#include "scene/framebuffer.h"
#include "figures/figures.h"
#include "scene/scene.h"
#include "textures/material.h"
#include "textures/texture.h"
#include <curand.h>
#include <curand_kernel.h>
#include "hlbvh/hlbvh.h"

#include "test.h"

bool cuda_main(HLBVH &bvh, Pixel *buffer, uint size, uint32_t &blocks_offset, \
               int subdiv_param, float max_height);
void cuda_load_to_gpu(Triangle *triangles, uint triangles_num,
                      Light lights[], uint lights_num,
                      Material &Material, Texture<float4> &texture, Texture<float> &displace);
void cuda_free_context();

void __global__
kernel_raytr(curandState_t *randbuff, Pixel *gpu_accum, uint32_t buff_size,
                Triangle *triangles,  uint32_t triangles_num,
                   Light *gpu_lights, uint32_t lights_num,
                Material *material,
              Texture<float4> *texture,
              Texture<float> *displace,
                   HLBVH bvh,
                uint32_t *stackMemory,
                uint32_t stackSize, uint32_t blocks_offset,
             int subdiv_param, float max_height);

//-----------------------------------------------------------------

//void cuda_init_eye(float x, float y, float z);
void cuda_reinit_frame(uint32_t width, uint32_t height);
void cuda_free_context();
void cuda_clear_buffs();
void cuda_update_scene();
void build_aabb(AABB *aabb, uint32_t size, float max_displace);
//-----------------------------------------------------------------
__device__ uint2 indToxy(uint ind, uint width);
inline uint __device__ __host__ index(uint2 xy, uint width);
