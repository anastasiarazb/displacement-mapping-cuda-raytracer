#include "cuda_raytr.h"
#include "../cuda_app.h"
#include "../scene/scene.h"
#include "../gpu_stack.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm_output/glm_outp.h"
#include "../cuda_numerics/float3.h"
#include "../cuda_numerics/float4.h"
#include <float.h>
#include <stdio.h>
#include <curand.h>

#ifdef SMOOTH
#define SCREENTOWORLD screenToWorldRand
#else
#define SCREENTOWORLD(rand, ind, width, height) screenToWorld(ind, width, height)
#endif

#ifdef TEST
#define INTERSECTION  rayDisplacedTriIntersection1
#else

#ifdef SIMPLE_DISPLACE
#define INTERSECTION(ray, tri, displaces, subdiv_param, max_height, thread_idx)\
    rayDisplacedTriIntersection1(ray, tri, displaces, subdiv_param)
#else
#ifdef NO_DISPLACE
#define INTERSECTION(...) rayTriangleIntersection(ray, triangles[idTriangle])
#else
#define INTERSECTION  rayDisplacedTriIntersection
#endif
#endif
#endif

extern __constant__ struct Scene gpu_context;

__device__ float3 screenToWorldRand(curandState_t *rand, uint ind, int width, int height)
{
    uint2   xy1 = indToxy(ind, width);
    float rand1 = curand_uniform(&rand[ind]) - 0.5f;
    float rand2 = curand_uniform(&rand[ind]) - 0.5f;
    float2   xy = make_float2(xy1.x + rand1, xy1.y + rand2);
    //    uint2 xy = indToxy(ind, width);

    float x = (xy.x * 2.0) / width - 1.0f;
    float y = (xy.y * 2.0) / height - 1.0f; //Перевод в нормальные координаты NDC

    glm::vec4  clip = glm::vec4(x, y, gpu_context.projectionPlaneZ, 1);
    glm::mat4     M = glm::make_mat4(gpu_context.ScreenToWorld);
    glm::vec4 world = M * clip; //Координаты пиксела в мировых координатах
    return make_float3(world.x, world.y, world.z);
}

__device__ float3 screenToWorld(uint ind, int width, int height)
{
    uint2 xy = indToxy(ind, width);
    float x = (float(xy.x) * 2.0) / width - 1.0f;
    float y = (float(xy.y) * 2.0) / height - 1.0f; //Перевод в нормальные координаты NDC

    glm::vec4  clip = glm::vec4(x, y, gpu_context.projectionPlaneZ, 1);
    glm::mat4     M = glm::make_mat4(gpu_context.ScreenToWorld);
    glm::vec4 world = M * clip; //Координаты пиксела в мировых координатах
    return make_float3(world.x, world.y, world.z);
}


__device__
float4 LambertShading(InnerPoint hit_point,
                      const Light &light,
                      uint tri_num, const Material *material)
{
    float3 dist = light.center - hit_point.p;
    float3 l = norma(dist);
    float3 n = norma(hit_point.n);
//    return light.color * fmax(dot(l, n), 0.0f) / (fmax(1.0f, len(dist)/15));
    return light.color * light.intensity * material->getDiffuse(tri_num) * fmax(dot(l, n), 0.0f)/fmax(1.0f, len(dist)/ATTENUATION);
}

// return param t
RTIntersection __device__ rayTrace(HLBVH &bvh, GpuStack<uint32_t> &stack, const Ray ray, const Triangle *triangles, \
                                   uint32_t &nearest_tri_id, const Texture<float> &displaces,\
                                   int subdiv_param, float max_height, uint thread_idx) {
    //    res_t = INFINITY;
    stack.push(0);
    RTIntersection res_hit;
    while(!stack.empty()) {
        uint32_t top = stack.top();
        stack.pop();

        // дети коробки из стека
        for (int i = 0; i < 2; i++) {
            float3 aabbMax = bvh.aabbMax[top + i];
            float3 aabbMin = bvh.aabbMin[top + i];
            if (rayAABBIntersection(ray.org, ray.dir, aabbMax, aabbMin)) {
                int link = bvh.links[top + i];
                if (link == LEAF) {
                    for (int j = getLeftBound (bvh.ranges[top + i]); \
                         j < getRightBound(bvh.ranges[top + i]); ++j) {

                        uint32_t idTriangle = bvh.references[j];
#ifdef DIFFERENCE
#ifdef SINGLE_KERNEL
                        //                        if (idTriangle != 928u) continue; //204457, tri_id = 928  192453
                        //                        if (idTriangle != 2511u) continue;
#endif
#endif
                        RTIntersection hit = INTERSECTION(ray, triangles[idTriangle], displaces, \
                                                          subdiv_param, max_height, thread_idx);
#ifdef DIFFERENCE
                        RTIntersection hit1 = rayTriangleIntersection(ray, triangles[idTriangle]);
                        if (hit1.success && !hit.success) {
                            hit = hit1;
                            printf("[rayTrace]: hit1 != hit2, id = %u, tri_id = %u; ray = (%f, %f, %f), (%f, %f, %f), tri = (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", \
                                   thread_idx, idTriangle, COORDS(ray.org), COORDS(ray.dir), COORDS(triangles[idTriangle].p0), COORDS(triangles[idTriangle].p1), COORDS(triangles[idTriangle].p2));
                        } else {
                            hit.success = false;
                        }
#endif
                        if (hit.success && hit.t < res_hit.t) {
                            res_hit = hit;
                            nearest_tri_id  = idTriangle;
                        }
                    } //for j from [getLeftBound (bvh.ranges[top + i]); getRightBound(bvh.ranges[top + i]))
                } else {
                    stack.push(bvh.links[top + i]);
                } //if (link == LEAF)
            } // if (rayAABBIntersection(ray.origin, ray.dir, aabbMax, aabbMin))
        } // for children of the aabb
    } //while(!stack.empty())

    return res_hit;
}

void __global__ kernel_raytr(curandState_t *randbuff,
                             Pixel *gpu_accum,  uint32_t buff_size,
                             Triangle *triangles,  uint32_t triangles_num,
                             Light *gpu_lights, uint32_t lights_num,
                             Material *material,
                             Texture<float4> *texture,
                             Texture<float> *displace,
                             HLBVH  bvh,
                             uint32_t *stackMemory, uint32_t stackSize, uint32_t blocks_offset,
                             int subdiv_param, float max_height) //адреса на VRAM
{
    uint id = threadIdx.x + (blockIdx.x + blocks_offset) * blockDim.x;
    if (id >= buff_size) {
        return;
    }
    //    if (id < gpu_context.width*(gpu_context.height/2 - 1)+gpu_context.width/2 + 90) return; //для subdiv_param = 5
//        if (id < gpu_context.width*(gpu_context.height/2-49)+gpu_context.width/2-10) return; //для subdiv_param = 5
#ifdef SINGLE_KERNEL
        if (id != gpu_context.width*(gpu_context.height/2-49)+gpu_context.width/2-10) return;
//    if (id != gpu_context.width*(gpu_context.height/2 +12)+gpu_context.width/2 + 36) return;
    //    if (id != 203657u) return;
    //    if (id != 192453u) return;
    //    if (id != 97821u) return;
//    if (id != 162098u) return;
    printf("------------------------------------\n");
#endif
    GpuStack<uint32_t> stack(&stackMemory[id * stackSize], stackSize);
    //Преобразование экранных координат пиксела к мировым
    float3 pixel = SCREENTOWORLD(randbuff, id, gpu_context.width, gpu_context.height);
    float3 start = gpu_context.eye;
    float3   dir = norma(pixel - start);
    uint32_t nearestTriIdx = triangles_num;
    Ray primary_ray(start, dir, 0.0f, INFINITY);
    //Первичная видимость: нахождение ближайшей видимой точки к экрану
    RTIntersection hit_params = rayTrace(bvh, stack, primary_ray, triangles, nearestTriIdx, \
                                         *displace, subdiv_param, max_height, id);
    if (hit_params.success) { //i_min < triangles_num
        InnerPoint P = hit_params.intersectionPoint;
        //Добавление компоненты фонового освещения, позволяющего видеть границы объекта
        //        printf("[kernel_raytr]: id = %u, P.uv = (%f, %f)\n", id, P.uv.x, P.uv.y);
//        if (P.uv.x > 1.0f || P.uv.y > 1.0f || P.uv.x < 0.0f || P.uv.y < 0.0f)
//            printf("[kernel_raytr]: id = %u, incorrect uv = (%f, %f)\n", id, P.uv.x, P.uv.y);
        float4 color = texture->get(P.uv);
#ifdef NORMAL_AS_COLOR
        float3 n = norma(hit_params.intersectionPoint.n);
        color = make_float4(n*0.5 + make_float3(0.5f));
#endif
        //Вторичный луч от заданной точки в источник света
        float3 biased_hit = start + norma(dir) * (hit_params.t - 0.01f);
        for (uint32_t  l = 0; l < lights_num; ++l) {
            float3 light_center = gpu_lights[l].center;
            //                     dir = light_center - hit;
            dir = light_center - biased_hit;
            //            float3 biased_hit = hit + norma(dir) * 0.01f; //Сдвиг необходим для исключения самопересечения
            Ray secondary_ray(biased_hit, dir, 0.0f, 1.0f);
            uint32_t blank = triangles_num;
            hit_params = rayTrace(bvh, stack, secondary_ray, triangles, blank, \
                                  *displace, subdiv_param, max_height, id);
            if (!hit_params.success) {
                //Пересечений нет: добавить компоненту диффузного освещения
                color += LambertShading(P, gpu_lights[l], nearestTriIdx, material);
            } //if
        } //for all lights
        // Запись полученного значения цвета в выходной буфер
#ifdef SMOOTH
        gpu_accum[id].safeAddColor(color, gpu_context.scale);
#else
        //        printf("[kernel_raytr]: id = %d, res color = (%f, %f, %f)\n", id, color.x, color.y, color.z);
        gpu_accum[id].set(color);
        //        printf("[kernel_raytr]: END addr: %p, id = %d, intersection = %d, color = (%f, %f, %f)\n", gpu_accum, id, gpu_accum[id].r, gpu_accum[id].g, gpu_accum[id].b);
#endif
    } else {
        visualize_lights(pixel, gpu_accum, id, gpu_lights, lights_num);
        return;
    } //if (hit_params.success)
}

