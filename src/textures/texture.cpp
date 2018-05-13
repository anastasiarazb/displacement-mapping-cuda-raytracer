#include "texture.h"
#include <stdio.h>
#include <stdlib.h>
#include <QImage>

template <>
float4 Texture<float4>::toTexelType(QColor color)
{
    return make_float4(color.redF(), color.greenF(), color.blueF(), color.alphaF());
}

template <>
float3 Texture<float3>::toTexelType(QColor color)
{
    return make_float3(color.redF(), color.greenF(), color.blueF());
}

template <>
float Texture<float>::toTexelType(QColor color)
{
    return color.lightnessF();
}


template <>
void Texture<float>::colorTransform(float shift, float scale)
{
    if (shift != 0.0f || scale != 1.0f) {
        for (size_t i = 0; i < size; ++i) {
            cpu_texels[i] = scale * (shift + cpu_texels[i]);
        }
        return;
    }
}
