#include "cuda_raytr.h"
#include "../cuda_numerics/float3.h"
#include "../cuda_numerics/float4.h"

extern __constant__ struct Scene gpu_context;

//Возвращает параметр отдаленности пересечения от начальной точки. Если < 0 - пересечение позади камеры или отсутствует, >= 0 - корректный параметр
__device__
float prime_intersection(const Light &S, const float3 &start, const float3 &dir)
{
    const float3 c = S.center;
    const float3 d = norma(dir);
    const float3 o = start;
    const float3 o_c = o-c;

    float B = dot(d, o_c);
    float A = lenSqr(d);
    float C = lenSqr(o_c) - S.Radius*S.Radius;
    float D = B*B - A*C;
    if (D < 0) return -1;
    if (D == 0) return -B/A;
    float sqrtD = sqrt(D);
    float t1 = (-B - sqrtD)/A;
    float t2 = (sqrtD - B)/A;
    float t_nearest = fmin(t1, t2);

    return t_nearest; //Шар может оказаться перед плоскостью экрана
}

void __device__ visualize_lights(float3 pixel, Pixel *gpu_accum, uint thread_id,
                                Light *gpu_lights, uint32_t lights_num)          //адреса на VRAM
{
    float3 start = gpu_context.eye;
    float3 dir   = pixel - start;
    Light  light;
    uint32_t i_min;
    float tmin = INFINITY;
    float t;
    for (int i = 0; i < lights_num; ++i) {
        i_min = lights_num;
        tmin  = INFINITY;
        light = gpu_lights[i];
        t = prime_intersection(light, start, dir);
        if (t >= 0 && t < tmin) {
            tmin  = t;
            i_min = i;
        }
        if (i_min < lights_num) {
            float4 color = gpu_lights[i_min].color * gpu_lights[i_min].intensity;
#ifdef SMOOTH
            gpu_accum[thread_id].safeAddColor(color, gpu_context.scale);
#else
            gpu_accum[thread_id].set(color);
#endif
            return;
        }

    }
}
