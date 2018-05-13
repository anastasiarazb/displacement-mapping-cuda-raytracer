#ifndef TEXTURE_H
#define TEXTURE_H
#include <cuda_runtime.h>
#include <QImage>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cmath>
#include "../cuda_err.h"
//#include <GLFW/glfw3.h>

template <typename T>
struct Texture
{
    size_t  size   = 0;
    size_t  height = 0;
    size_t  width  = 0;

    bool enabled = true;

    T  *cpu_texels;
    T  *gpu_texels;

    __device__
    T get(float u, float v) const;
    __device__
    T get(float2 uv) const {return get(uv.x, uv.y);}

    T cpu_get(float u, float v) const;
    T cpu_get(float2 uv) const {return cpu_get(uv.x, uv.y);}

    void initTexture();
    void setColor(float r = 1.0, float g = 1.0, float b = 1.0, float alpha = 1.0);
    void destroy();
    bool load(const char *path);
    void sendCopyToGPU(Texture<T> *gpu_copy);
    void colorTransform(float shift, float scale);

    T toTexelType(QColor color);
};

//------------------------------------------------

template <>
void Texture<float>::colorTransform(float shift, float scale);

template <typename T>
__device__
T Texture<T>::get(float u, float v) const
{
    if (u < 0.0f || v < 0.0f || u > 1.0f || v > 1.0f) {
        if (u < 0.0f) u = 0.0f;
        if (v < 0.0f) v = 0.0f;
        if (u > 1.0f) u = 1.0f;
        if (v > 1.0f) v = 1.0f;
//        printf("[Texture::get]: incorrect parameters u, v = (%f, %f) not in [0, 1]\n", u, v);
    }
//    uint32_t x = (uint32_t)round(u * float(width-1));
//    uint32_t y = (uint32_t)round(v * float(height-1));
//    return gpu_texels[y * width + x];
    //Линейная интерполяция

    float s = u * float(width-1);
    float t = v * float(height-1);

    float floor_s = floor(s);
    float floor_t = floor(t);

    uint32_t x0 = (uint32_t)floor_s;
    uint32_t y0 = (uint32_t)floor_t;
    uint32_t x1 = fmin(ceil(s), float(width-1));
    uint32_t y1 = fmin(ceil(t), float(height-1));

    T v00 = gpu_texels[y0 * width + x0];
    T v01 = gpu_texels[y0 * width + x1];
    T v10 = gpu_texels[y1 * width + x0];
    T v11 = gpu_texels[y1 * width + x1];

    //Оставить только дробные части
    s -= floor_s;
    t -= floor_t;

    return (1.f - s)*(1.f - t)*v00 + (1.f - s)*t*v01 + s * (1.f - t)*v10 + s * t * v11;
}

template <typename T>
void Texture<T>::sendCopyToGPU(Texture<T> *gpu_copy)
{
    gpuErrchk(cudaMalloc(&gpu_texels, size * sizeof(T)));
    gpuErrchk(cudaMemcpy(gpu_texels, cpu_texels, size * sizeof(T), cudaMemcpyHostToDevice));

    Texture<T> temp; //Ее копия отправится на GPU. Саму копию не надо будет освобождать специально, так как там только указатели на память на GPU и на CPU, которые хранятся и освобождаются явно в базовом объекте, который остается на CPU
    temp.cpu_texels = nullptr;
    temp.gpu_texels = gpu_texels;
    temp.size   = size;
    temp.width  = width;
    temp.height = height;
    if (gpu_copy == NULL) {
        gpuErrchk(cudaMalloc(&gpu_copy, sizeof(Texture)));
    }
    gpuErrchk(cudaMemcpy(gpu_copy, &temp, sizeof(Texture), cudaMemcpyHostToDevice)); //Отправить структуру
}

template <typename T>
void Texture<T>::destroy()
{
    if (!cpu_texels) {
        free(cpu_texels);
        cpu_texels = nullptr;
        printf("[Texture::destroy]: free cpu_texels\n");
    }
    if (!gpu_texels) {
        gpuErrchk(cudaFree(gpu_texels));
        gpu_texels = nullptr;
        printf("[Texture::destroy]: cudaFree gpu_texels\n");
    }
    initTexture();
}

template <typename T>
T Texture<T>::cpu_get(float u, float v) const
{
    uint32_t x = (uint32_t)round(u * float(width-1));
    uint32_t y = (uint32_t)round(v * float(height-1));
    return   cpu_texels[y * width + x];
}

template <typename T>
void Texture<T>::initTexture()
{
    size   = 0;
    height = 0;
    width  = 0;
    cpu_texels = nullptr;
    gpu_texels = nullptr;
}

template <typename T>
void Texture<T>::setColor(float r, float g, float b, float alpha)
{
    cpu_texels = (T *)malloc(sizeof(T));
    size = 1;
    width = 1;
    height = 1;
    cpu_texels[0] = toTexelType(QColor(int(r*255), int(g*255), int(b*255), int(alpha*255)));
}

// --------------------------
template <typename T>
T Texture<T>::toTexelType(QColor color)
{
    return T(color);
}

template <>
float4 Texture<float4>::toTexelType(QColor color);

template <>
float3 Texture<float3>::toTexelType(QColor color);

template <>
float Texture<float>::toTexelType(QColor color);
// --------------------------
template <typename T>
bool Texture<T>::load(const char *path)
{
    //QImage qim(QString(path), Q_NULLPTR);
    QImage qim;
    if (path == nullptr || !qim.load(QString(path))) {
        cpu_texels = nullptr;
        return false;
    }
    width  = qim.width();
    height = qim.height();
    size = width * height;
    cpu_texels = (T *)malloc(size * sizeof(T));
    for (size_t i = 0; i < size; ++i) {
        QColor color = qim.pixelColor(i % width, height - 1 - i / width);
        cpu_texels[i] = toTexelType(color);
    }
    return size > 0;
}

#endif // TEXTURE_H
