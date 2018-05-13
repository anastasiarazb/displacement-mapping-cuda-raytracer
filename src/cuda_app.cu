#include "cuda_app.h"
#include "scene/scene.h"
#include "hlbvh/hlbvh.h"
#include "textures/texture.h"
#include "gpu_buffs.h"
#include "kernel_params.h"
#include <stdlib.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm_output/glm_outp.h"
#include "cuda_numerics/float4.h"
#include "cuda_err.h"
#include <curand.h>
#include <curand_kernel.h>
#include "main.h"

#define N 2

GPU_buffs gpu_buffs;
Material *gpu_material;
Texture<float4>  *gpu_texture;
Texture<float>  *gpu_displace;
Light    *gpu_lights;
Triangle *gpu_triangles;
uint   global_triangles_num;
uint   global_lights_num;

__constant__ struct Scene gpu_context;
extern Scene scene;

//--------------------------------------------------------------

__host__ __device__
unsigned int countThreads(unsigned int size)
{
    int x = size/32 + 1;
    return x * 32;
}


__host__ __device__
unsigned int countBlocks(unsigned int size)
{
    return (size + BLOCKSIZE - 1)/BLOCKSIZE;
}

//--------------------------------------------------------------

inline uint __device__ __host__ index(uint2 xy, uint width)
{
    return xy.x + xy.y * width;
}

__device__ uint2 indToxy(uint ind, uint width)
{
    return make_uint2(ind % width, ind / width);
}

//--------------------------------------------------------------

void cuda_reinit_frame(uint32_t width, uint32_t height)
{
    scene.reinitScene(width, height);
    scene.sendToGPU();
    gpu_buffs.reinitFrame(width, height);
}

void cuda_update_scene()
{
    gpu_buffs.clearBuffs();
    scene.update();
}

void cuda_clear_buffs()
{
    gpu_buffs.clearBuffs();
}

//--------------------------------------------------------

__global__
void kernel_cudaScale(Pixel *accum, Pixel *buff, uint buf_size, size_t counter, uint32_t blocks_offset)
{
    uint id = threadIdx.x + (blockIdx.x + blocks_offset) * blockDim.x;
    if (id >= buf_size) {
        return;
    }

    float4 averaged = accum[id].vec4 / (float)counter;
    buff[id].vec4   = averaged / gpu_context.scale;

//    if (id == (600*400)) {
//        printf("[cuda_app.cu/kernel_cudaScale]: id = %u, SUCCESS\n", id);
//    }
}
//----------------------------------------------------------------------------------
void Scene::sendToGPU()
{
    gpuErrchk(cudaMemcpyToSymbol(gpu_context, this, sizeof(Scene), 0, cudaMemcpyHostToDevice));
}

void cuda_load_to_gpu(Triangle *triangles, uint triangles_num,
                      Light lights[], uint lights_num,
                      Material &material, Texture<float4> &texture, Texture<float> &displace)
{
    global_triangles_num = triangles_num;
    gpuErrchk(cudaMalloc(&gpu_triangles, triangles_num * sizeof(Triangle)));
    gpuErrchk(cudaMemcpy( gpu_triangles, triangles, triangles_num * sizeof(Triangle), cudaMemcpyHostToDevice));
    printf("[cuda_load_to_gpu]: gpu_triangles: %u elems copied to %p\n", triangles_num, gpu_triangles);

    global_lights_num = lights_num;
    gpuErrchk(cudaMalloc(&gpu_lights, lights_num * sizeof(Light)));
    gpuErrchk(cudaMemcpy( gpu_lights, lights, lights_num * sizeof(Light), cudaMemcpyHostToDevice));
    printf("[cuda_load_to_gpu]: global_lights: %u elems copied\n", lights_num);

    gpuErrchk(cudaMalloc(&gpu_material, sizeof(Material)));
    material.sendCopyToGPU(gpu_material);
    printf("[cuda_load_to_gpu]: gpu_material: sendCopyToGPU() performed\n");

    gpuErrchk(cudaMalloc(&gpu_texture, sizeof(Texture<float4>)));
    texture.sendCopyToGPU(gpu_texture);
    printf("[cuda_load_to_gpu]: gpu_texture: sendCopyToGPU() performed\n");
    gpuErrchk(cudaMalloc(&gpu_displace, sizeof(Texture<float>)));
    displace.sendCopyToGPU(gpu_displace);
    printf("[cuda_load_to_gpu]: gpu_displaced: sendCopyToGPU() performed\n");

//    printf("[cuda_app.cu/cuda_load_to_gpu]: triangles_num = %u, lights_num = %u and Material loaded to GPU\n",
//           triangles_num, lights_num);
}

// -----------------------------------------------

__global__
void computeAABB(AABB *aabb, Triangle *gpu_triangles, uint32_t buf_size, float max_displace) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= buf_size) {
        return;
    }
    Triangle down, up, medium;
    medium = gpu_triangles[id];
    float3 N0 = max_displace * medium.n0;
    float3 N1 = max_displace * medium.n1;
    float3 N2 = max_displace * medium.n2;
    down.set(medium.p0 - N0, medium.p1 - N1, medium.p2 - N2);
      up.set(medium.p0 + N0, medium.p1 + N1, medium.p2 + N2);
    getRightBound(aabb[id].x) = fmax(down.p0.x, down.p1.x, down.p2.x,
                                       up.p0.x,   up.p1.x,   up.p2.x);
    getLeftBound (aabb[id].x) = fmin(down.p0.x, down.p1.x, down.p2.x,
                                       up.p0.x,   up.p1.x,   up.p2.x);

    getRightBound(aabb[id].y) = fmax(down.p0.y, down.p1.y, down.p2.y,
                                       up.p0.y,   up.p1.y,   up.p2.y);
    getLeftBound (aabb[id].y) = fmin(down.p0.y, down.p1.y, down.p2.y,
                                       up.p0.y,   up.p1.y,   up.p2.y);

    getRightBound(aabb[id].z) = fmax(down.p0.z, down.p1.z, down.p2.z,
                                       up.p0.z,   up.p1.z,   up.p2.z);
    getLeftBound (aabb[id].z) = fmin(down.p0.z, down.p1.z, down.p2.z,
                                       up.p0.z,   up.p1.z,   up.p2.z);
}

void build_aabb(AABB *aabb, uint32_t size, float max_displace) {
    dim3 threadsInBlock(BLOCKSIZE, 1, 1);
    dim3 numblocks(countBlocks(size));
    computeAABB<<<numblocks, threadsInBlock>>>(aabb, gpu_triangles, size, max_displace);
}

void cuda_free_context()
{
    gpu_buffs.destroy();
    //gpu_material: Удаляется только копия на gpu, буфферы освобождаются через cpu_buffer.destroy()
    gpuErrchk(cudaFree(gpu_material));
    gpuErrchk(cudaFree(gpu_texture));
    gpuErrchk(cudaFree(gpu_displace));
    gpuErrchk(cudaFree(gpu_lights));
    gpuErrchk(cudaFree(gpu_triangles));
    printf("[cuda_free_context]: free: gpu_material, gpu_texture, gpu_displace, gpu_lights, gpu_triangles\n");
}

//--------------------------------------------------------
/*
 * Заполнение буфера кадра.
 * Возвращает true, если построение кадра прошло успешно и false иначе.
 * Аргументы:
 * HLBVH &bvh    - дерево ограничивающих оболочек элементов модели
 * Pixel *buff   - буфер кадра
 * uint buf_size - размер массива под буфер кадра
 * uint32_t &blocks_offset - часть кадра, которая должна быть отрисована
 * int subdiv_param - параметр разбиения треугольников при наложении карты смещений
 * float max_height - максимальный сдвиг при наложении карты смещений
 */
bool cuda_main(HLBVH &bvh, Pixel *buff, uint buf_size, uint32_t &blocks_offset,
               int subdiv_param, float max_height)
{
    GpuStackAllocator::getInstance().pushPosition(); //Стек для обхода дерева оболочек
    dim3 threadsInBlock(BLOCKSIZE, 1, 1); //Установка количества потоков в блоке, размерность блока 1х1

    if (blocks_offset >= countBlocks(buf_size)) {
        scene.frames_counter += 1; //Количество отрисованных кадров с начала работы приложения
        blocks_offset = 0;
    }

    // Размер одного GPU стека
    uint32_t stackSize = (bvh.numBVHLevels * 2 + 1);
    // Память под GPU стек для каждого потока
    uint32_t *stackMemory = GpuStackAllocator::getInstance().alloc<uint32_t>(buf_size * stackSize);
    kernel_raytr<<<blocks_per_frame, threadsInBlock>>> \
                                           (gpu_buffs.rand_buff, gpu_buffs.accum_pixels, buf_size, \
                                            gpu_triangles, global_triangles_num, gpu_lights, global_lights_num, \
                                            gpu_material, gpu_texture, gpu_displace, bvh, stackMemory, stackSize, blocks_offset,
                                            subdiv_param, max_height);
    if (gpuErrchk(cudaGetLastError())) {
        return false;
    }
#ifdef SMOOTH
    //Вычисление среднего арифметического значения цвета по всем кадрам
    kernel_cudaScale<<<blocks_per_frame, threadsInBlock>>> \
                                                  (gpu_buffs.accum_pixels, gpu_buffs.res_pixels, \
                                                   buf_size, scene.frames_counter, blocks_offset);
    if (gpuErrchk(cudaGetLastError())) {
        return false;
    }
    //Копирование усредненного буфера кадра на CPU
    gpuErrchk(cudaMemcpy(buff, gpu_buffs.res_pixels, buf_size * sizeof(Pixel), \
                         cudaMemcpyDeviceToHost));
#else
    //Копирование последнего полученного буфера кадра на CPU
    gpuErrchk(cudaMemcpy(buff, gpu_buffs.accum_pixels, buf_size * sizeof(Pixel), \
                         cudaMemcpyDeviceToHost));
#endif
    GpuStackAllocator::getInstance().popPosition();
    blocks_offset += blocks_per_frame;
    cudaDeviceSynchronize();
    return true;
}
