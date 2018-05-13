#include "material.h"
#include "malloc.h"
#include "../cuda_numerics/float4.h"
#include "../test.h"

#define DEBUG

unsigned int pitch256(int num, size_t size_of_type)
{
    size_t x = ((255 + num*size_of_type)/256); //Длина буфера для массива элементов type, выровненная по 256 байт
    return x * 256;
}

void Material::initMaterial(int num = 2)
{
    this->num = num;
    unsigned int float_buff_size = pitch256(num, sizeof(float));
    unsigned int color_buff_size = pitch256(num, sizeof(float4));
    unsigned int size = float_buff_size * 3 + color_buff_size;
    cpu_buff = (char *)calloc(size, 1); //Пусть будет забито нулями на всякий случай
    gpu_buff = nullptr; //Память выделяется в sendCopyToGPU
    ambient = (float *) cpu_buff;
    diffuse = (float *)(cpu_buff + float_buff_size);
    reflect = (float *)(cpu_buff + 2 * float_buff_size);
    color  = (float4 *)(cpu_buff + 3 * float_buff_size);

    diffuse[0] = DIFF;
    ambient[0] = AMBIENT;
    reflect[0] = 1.f;
    color[0] = make_float4(0.f, 1.f, 0.f);

    diffuse[1] = DIFF;
    ambient[1] = AMBIENT;
    reflect[1] = 1.f;
    color[1] = make_float4(0.f, 1.f, 1.f);
}
