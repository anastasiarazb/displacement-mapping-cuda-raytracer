#include "material.h"
#include "../cuda_err.h"
#include "../cuda_numerics/float4.h"

void Material::sendCopyToGPU(Material *gpu_texture) //Хранится на CPU, память выделяет на GPU
{
    unsigned int float_buff_size = pitch256(num, sizeof(float));
    unsigned int color_buff_size = pitch256(num, sizeof(float4));
    unsigned int size = float_buff_size * 3 + color_buff_size;
    gpuErrchk(cudaMalloc(&gpu_buff, size));
    gpuErrchk(cudaMemcpy(gpu_buff, cpu_buff, size, cudaMemcpyHostToDevice));
    Material temp; //Ее копия отправится на GPU. Саму копию не надо будет освобождать специально, так как там только указатели на память на GPU и на CPU, которые хранятся и освобождаются явно в базовом объекте, который остается на CPU
    temp.num = num;
    temp.gpu_buff = gpu_buff;
    temp.cpu_buff = NULL;
    temp.ambient = (float *)gpu_buff;
    temp.diffuse = (float *)(gpu_buff + float_buff_size);
    temp.reflect = (float *)(gpu_buff + 2 * float_buff_size);
    temp.color = (float4 *)(gpu_buff + 3 * float_buff_size);
    if (gpu_texture == NULL) {
        gpuErrchk(cudaMalloc(&gpu_texture, sizeof(Material)));
    }
    gpuErrchk(cudaMemcpy(gpu_texture, &temp, sizeof(Material), cudaMemcpyHostToDevice)); //Отправить структуру
}

void Material::destroy()
{
    if (cpu_buff != NULL)
    {
        free(cpu_buff);
        cpu_buff = NULL;
    }
    if (gpu_buff != nullptr)
    {
        gpuErrchk(cudaFree(gpu_buff));
        gpu_buff = NULL;
    }
}

FUNC_PREFIX
float Material::getAmbient(unsigned int i) const
{
    return ambient[i % num];
}

FUNC_PREFIX
float4 Material::getColor(unsigned int i) const
{
    return color[i % num];
}

FUNC_PREFIX
float Material::getDiffuse(unsigned int i) const
{
    return diffuse[i % num];
}

FUNC_PREFIX
float Material::getReflect(unsigned int i) const
{
    return reflect[i % num];
}
