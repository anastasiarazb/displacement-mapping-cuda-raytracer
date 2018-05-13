//#include "texture.h"
//#include "../cuda_err.h"
//#include <inttypes.h>
//#include <stdio.h>

//template <typename T>
//__device__
//T Texture<T>::get(float u, float v)
//{
//    uint32_t x = round(u * float(width-1));
//    uint32_t y = round(v * float(height-1));
//    return gpu_texels[y * width + x];
//}

//template <typename T>
//void Texture<T>::sendCopyToGPU(Texture<T> *gpu_copy)
//{
//    gpuErrchk(cudaMalloc(&gpu_texels, size * sizeof(T)));
//    gpuErrchk(cudaMemcpy(gpu_texels, cpu_texels, size * sizeof(T), cudaMemcpyHostToDevice));

//    Texture<T> temp; //Ее копия отправится на GPU. Саму копию не надо будет освобождать специально, так как там только указатели на память на GPU и на CPU, которые хранятся и освобождаются явно в базовом объекте, который остается на CPU
//    temp.cpu_texels = nullptr;
//    temp.gpu_texels = gpu_texels;
//    temp.size   = size;
//    temp.width  = width;
//    temp.height = height;
//    if (gpu_copy == NULL) {
//        gpuErrchk(cudaMalloc(&gpu_copy, sizeof(Texture)));
//    }
//    gpuErrchk(cudaMemcpy(gpu_copy, &temp, sizeof(Texture), cudaMemcpyHostToDevice)); //Отправить структуру
//}

//template <typename T>
//void Texture<T>::destroy()
//{
//    if (!cpu_texels) {
//        free(cpu_texels);
//        cpu_texels = nullptr;
//        printf("[Texture::destroy]: free cpu_texels\n");
//    }
//    if (!gpu_texels) {
//        gpuErrchk(cudaFree(gpu_texels));
//        gpu_texels = nullptr;
//        printf("[Texture::destroy]: cudaFree gpu_texels\n");
//    }
//    initTexture();
//}
