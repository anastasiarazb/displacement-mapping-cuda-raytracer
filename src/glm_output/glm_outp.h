#pragma once
#include <glm/glm.hpp>
#include <iostream>
#include "cuda_runtime.h"

std::ostream& operator<< (std::ostream& os, const glm::vec4& v);
std::ostream& operator<< (std::ostream& os, const glm::vec3& v);
std::ostream& operator<< (std::ostream& os, const glm::mat4& M);

__host__ __device__ void printfloat3(const float3 &v);
__host__ __device__ void printfloat3e(const float3 &v);
__host__ __device__ void printfloat4e(const float4 &v);
__host__ __device__ void printuchar3(const uchar3 &v);
__host__ __device__ void printv3f(const glm::vec3 &v);
__host__ __device__ void printv4f(const glm::vec4 &v);
__host__ __device__ void print4x4f(const glm::mat4 &M);


float3 tofloat3(const glm::vec3 &v);
void   matrCopy(float dst[], const glm::mat4 &src);
