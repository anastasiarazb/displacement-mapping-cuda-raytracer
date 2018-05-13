#pragma once

#include <cuda_runtime.h>
#define FUNC_PREFIX __host__ __device__
//#include "figures/figures.h"

class Material //Пока заглушки. Будут методы для загрузки текстур
{
  float4 *color;
  float *ambient;
  float *diffuse;
  float *reflect;

  unsigned int num;
  char *cpu_buff;
  char *gpu_buff;

public:

  void initMaterial(int num); //malloc
  void sendCopyToGPU(Material *gpu_texture); //cudaMalloc
  void destroy(); //free, cudaFree
  FUNC_PREFIX float getAmbient(unsigned int i) const;
  FUNC_PREFIX float4 getColor(unsigned int i) const;
  FUNC_PREFIX float getDiffuse(unsigned int i) const;
  FUNC_PREFIX float getReflect(unsigned int i) const;
};

unsigned int pitch256(int num, size_t size_of_type);
