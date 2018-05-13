#include "figures/figures.h"
#include "scene/callbacks.h"
#include "cuda_app.h"
#include "scene/scene.h"
#include "cuda_numerics/float3.h"
#include "cuda_numerics/int3.h"
#include "textures/material.h"
#include "textures/texture.h"
#include "parsers/obj.h"
#include "model/model.h"
#include "hlbvh/hlbvh.h"
#include <string>
#include <stdio.h>

#include "test.h"

#ifdef TEST
enum class RayToPoint {
    LEFT, RIGHT, FORWARD
};

enum class IntersectionType{
    ENTER, EXIT, NONE
};

//FUNC_PREF
//const char * toString(RayToPoint a)
//{
//    switch (a) {
//    case RayToPoint::FORWARD:
//             return "FORWARD";
//    case RayToPoint::LEFT:
//             return "LEFT";
//    case RayToPoint::RIGHT:
//             return "RIGHT";
//    default:
//        return "[RayToPointToString]: ERROR: undefined type.\n";
//    }
//}

//FUNC_PREF
//const char * toString(IntersectionType a)
//{
//    switch (a) {
//    case IntersectionType::ENTER:
//                   return "ENTER";
//    case IntersectionType::EXIT:
//                   return "EXIT";
//    case IntersectionType::NONE:
//                   return "NONE";
//    default:
//        return "[IntersectionTypeToString]: ERROR: undefined type.\n";
//    }
//}

//---------------------------------------------------------------------------
__device__
RayToPoint rayToPoint(const Ray &ray, const InnerPoint &P)
{
    float sign = dot(ray.dir, cross(P.n, ray.origin - P.p));
    return sign == 0.0 ? RayToPoint::FORWARD : (sign < 0? RayToPoint::LEFT : RayToPoint::RIGHT);
}

__device__
//Поиск параметра в предположении, что луч проходит справа от A и слева от B
int sideBinSearchInner(const Ray &ray, InnerPoint A, InnerPoint B, int subdiv_param)
{
    int low  = 0;
    int high = subdiv_param;
    InnerPoint pivot;
    while (high - low > 1) {
        int med = (high + low)/2;
        pivot.setInterpolate(A, B, float(med)/float(subdiv_param));
        RayToPoint side = rayToPoint(ray, pivot);
        if (side == RayToPoint::LEFT) {
            high = med;
        } else {
            low = med;
        }
    }
    return low;
}

__device__
IntersectionType rayTopBottomAttitude(const Ray &ray, const Triangle &triangle, \
                                      int subdiv_param, int3 *micro_tri_index)
{
    RTIntersection intersection = rayTriangleIntersection(ray, triangle);
    if (!intersection.success) {
        micro_tri_index->x = -1;
        micro_tri_index->y = -1;
        micro_tri_index->z = -1;
        return IntersectionType::NONE;
    } else if (dot(ray.origin, triangle.normal()) < 0) { //Нормаль и луч разнонаправлены => луч входит
        micro_tri_index->x = intersection.coords.alpha()*subdiv_param;
        micro_tri_index->y = intersection.coords.beta  *subdiv_param;
        micro_tri_index->z = intersection.coords.gamma *subdiv_param;
        if (micro_tri_index->x + micro_tri_index->y + micro_tri_index->z > subdiv_param - 1) {
            printf("[rayTopBottomAttitude]: i + j + k = %d + %d + %d = %d, a-beta-gamma=(%f, %f, %f)\n",\
                   micro_tri_index->x, micro_tri_index->y, micro_tri_index->z, \
                   micro_tri_index->x + micro_tri_index->y + micro_tri_index->z, \
                   intersection.coords.alpha(), intersection.coords.beta, intersection.coords.gamma);
        }
        return IntersectionType::ENTER;
    } else {
        micro_tri_index->x = intersection.coords.alpha()*subdiv_param;
        micro_tri_index->y = intersection.coords.beta  *subdiv_param;
        micro_tri_index->z = intersection.coords.gamma *subdiv_param;
        if (micro_tri_index->x + micro_tri_index->y + micro_tri_index->z > subdiv_param - 1) {
            printf("[rayTopBottomAttitude]: i + j + k = %d + %d + %d = %d, a-beta-gamma=(%f, %f, %f)\n",\
                   micro_tri_index->x, micro_tri_index->y, micro_tri_index->z, \
                   micro_tri_index->x + micro_tri_index->y + micro_tri_index->z, \
                   intersection.coords.alpha(), intersection.coords.beta, intersection.coords.gamma);
        }
        return IntersectionType::EXIT;
    }
//    return IntersectionType::NONE;
}

__device__
IntersectionType raySideAttitude(const Ray &ray, InnerPoint A, InnerPoint B, int subdiv_param, int *micro_tri_param)
{
    RayToPoint rayA = rayToPoint(ray, A);
    RayToPoint rayB = rayToPoint(ray, B);
//    printf("[raySideAttitude]: A: %s, B: %s\n", toString(rayA), toString(rayB));s
    if (rayA == rayB) {
        *micro_tri_param = -1;
        return IntersectionType::NONE;
    }
    //Случай вхождения луча в призму через сторону AB: A-^-B
    if (rayA == RayToPoint::RIGHT) { //rayB == Side::LEFT || rayB == Side::Forward
            *micro_tri_param = sideBinSearchInner(ray, A, B, subdiv_param);
            return IntersectionType::ENTER;
    //Случай выхода луча из призмы через сторону AB: A-v-B (то же самое B-^-A)
    } else if (rayA == RayToPoint::LEFT) { //rayB == Side::Right || rayB == Side::Forward
            *micro_tri_param = subdiv_param - sideBinSearchInner(ray, B, A, subdiv_param) - 1;
            return IntersectionType::EXIT;
    } else if (rayA == RayToPoint::FORWARD) {
        if (rayB == RayToPoint::LEFT) { //A-^-B
            *micro_tri_param = sideBinSearchInner(ray, A, B, subdiv_param);
            return IntersectionType::ENTER;
        } else { //rayB == RayToPoint::RIGHT B-^-A
            *micro_tri_param = subdiv_param - sideBinSearchInner(ray, B, A, subdiv_param) - 1;
            return IntersectionType::EXIT;
        }
    }
    *micro_tri_param = -1;
    return IntersectionType::NONE; //Сюда не должно попадать управление, но того требует компилятор
}

//  с
//  |\     k^      i растет по диагонали от cnjhjys сb к a
//  a-b   j ->
__device__
void setUVabcByijk(Baricentric *uva, Baricentric *uvb, Baricentric *uvc,
                   int i, int j, int k, int subdiv_param){
    float delta = 1.f/float(subdiv_param);
    if (i + j + k == subdiv_param - 1) { //Нижний микротреугольник
        uva->set( j   *delta,  k   *delta);
        uvb->set((j+1)*delta,  k   *delta);
        uvc->set( j   *delta, (k+1)*delta);
        return;
    } else if (i + j + k == subdiv_param - 2) { //верхний микротреугольник
        uva->set((j+1)*delta,  k   *delta);
        uvb->set((j+1)*delta, (k+1)*delta);
        uvc->set( j   *delta, (k+1)*delta);
        return;
    } else {
        printf("[setUVabcByijk]: Error: i+j+k=%d+%d+%d != N-1 or N-2, N = %d\n",
               i, j, k, subdiv_param);
    }
}

//Луч должен входить со стороны ab
__device__
bool displaceInitPhase(const Ray &ray, const Triangle &base_tri, const Triangle top_tri, \
                       int3 *start_cell, int3 *end_cell,
                       Baricentric *uva, Baricentric *uvb, Baricentric *uvc, int subdiv_param)
{
    int3 bad_index = make_int3(-1, -1, -1);
    *start_cell = *end_cell = bad_index;
    int cell_index = -1;
    IntersectionType type;
    InnerPoint A, B;
    //Сторона P0P1 (j) - нижние микротреугольники => i + j + k = subdiv_param - 1, k = 0
    A = base_tri.getVertex0();
    B = base_tri.getVertex1();
    type = raySideAttitude(ray, A, B, subdiv_param, &cell_index);
    int enter_side = -1;
    if (type == IntersectionType::ENTER) {
        int i = subdiv_param - 1 - cell_index;
        int j = cell_index;
        int k = 0;
        *start_cell = make_int3(i, j, k);
        setUVabcByijk(uva, uvb, uvc, i, j, k, subdiv_param);
        enter_side = 0;
    } else if (type == IntersectionType::EXIT) {
        *end_cell   = make_int3(subdiv_param - 1 - cell_index, cell_index, 0);
    }

    //Сторона P1P2 (k в положительном направлении) - нижние микротреугольники => i + j + k = subdiv_param - 1, i = 0
    A = base_tri.getVertex1();
    B = base_tri.getVertex2();
    type = raySideAttitude(ray, A, B, subdiv_param, &cell_index);
    if (type == IntersectionType::ENTER) {
        int i = 0;
        int j = subdiv_param - 1 - cell_index;
        int k = cell_index;
        *start_cell = make_int3(i, j, k);
        setUVabcByijk(uvc, uva, uvb, i, j, k, subdiv_param);
        enter_side = 1;
    } else if (type == IntersectionType::EXIT) {
        *end_cell   = make_int3(0, subdiv_param - 1 - cell_index, cell_index);
    }

    if (*start_cell != bad_index && *end_cell != bad_index) {
        return true;
    }

    //Сторона P2P0 (k в отрицательном направлении) - нижние микротреугольники => i + j + k = subdiv_param - 1, j = 0
    A = base_tri.getVertex2();
    B = base_tri.getVertex0();
    type = raySideAttitude(ray, A, B, subdiv_param, &cell_index);
    cell_index = subdiv_param - 1 - cell_index;
    if (type == IntersectionType::ENTER) {
        int i = subdiv_param - 1 - cell_index;
        int j = 0;
        int k = cell_index;
        *start_cell = make_int3(i, j, k);
        setUVabcByijk(uvb, uvc, uva, i, j, k, subdiv_param);
        enter_side = 2;
    } else if (type == IntersectionType::EXIT) {
        *end_cell   = make_int3(subdiv_param - 1 - cell_index, 0, cell_index);
    }

    if (*start_cell != bad_index && *end_cell != bad_index) {
        return true;
    }

    int3 cell = bad_index;
    //Основание
    type = rayTopBottomAttitude(ray, base_tri, subdiv_param, &cell);
    if (type == IntersectionType::ENTER) {
        *start_cell = cell; //На какой-то итерации ранее, возможно, уже определена строна входа
        int i = cell.x;
        int j = cell.y;
        int k = cell.z;
        switch (enter_side) {
        case 0:
            setUVabcByijk(uva, uvb, uvc, i, j, k, subdiv_param);
            break;
        case 1:
            setUVabcByijk(uvc, uva, uvb, i, j, k, subdiv_param);
            break;
        case 2:
            setUVabcByijk(uvb, uvc, uva, i, j, k, subdiv_param);
            break;
        default: //Луч проходит треугольник насквозь - неважно, как определены стороны
            setUVabcByijk(uva, uvb, uvc, i, j, k, subdiv_param);
        }
    } else if (type == IntersectionType::EXIT) {
        *end_cell   = cell;
    }

    if (*start_cell != bad_index && *end_cell != bad_index) {
        return true;
    }

    //Крышка
    type = rayTopBottomAttitude(ray, top_tri, subdiv_param, &cell);
    if (type == IntersectionType::ENTER) {
        *start_cell = cell; //На какой-то итерации ранее, возможно, уже определена строна входа
        int i = cell.x;
        int j = cell.y;
        int k = cell.z;
        switch (enter_side) {
        case 0:
            setUVabcByijk(uva, uvb, uvc, i, j, k, subdiv_param);
            break;
        case 1:
            setUVabcByijk(uvc, uva, uvb, i, j, k, subdiv_param);
            break;
        case 2:
            setUVabcByijk(uvb, uvc, uva, i, j, k, subdiv_param);
            break;
        default:
            setUVabcByijk(uva, uvb, uvc, i, j, k, subdiv_param);
        }
    } else if (type == IntersectionType::EXIT) {
        *end_cell   = cell;
    }

    if (*start_cell != bad_index && *end_cell != bad_index) {
        return true;
    }

    return false;
}

enum class LastChange {
    iplus, jminus, kplus, iminus, jplus, kminus //0-5
};

__device__
RTIntersection rayDisplacedTriIntersection(const Ray &ray, const Triangle &triangle,
                                               const Texture<float> &displaces,
                                               int subdiv_param, float shift, float scale)
{
    int3 start, end;
    float h0 = scale * (shift + displaces.get(triangle.uv0));
    float h1 = scale * (shift + displaces.get(triangle.uv1));
    float h2 = scale * (shift + displaces.get(triangle.uv2));
    Triangle top = triangle;
    top.displace(h0, h1, h2);
    top.setDefaultNormals();

    RTIntersection miss;
    miss.success = false;
    Baricentric uva, uvb, uvc; //Координаты alpha-beta-gamma для каждой вершины
    if (!displaceInitPhase(ray, triangle, top, &start, &end, &uva, &uvb, &uvc, subdiv_param)) {
        return miss;
    }

    int i = start.x;
    int j = start.y;
    int k = start.z;

//    setUVabcByijk(&uva, &uvb, &uvc, i, j, k, subdiv_param);

    InnerPoint a, b, c; //текущий микротреугольник
    a = triangle.interpolate(uva);
    b = triangle.interpolate(uvb);
    c = triangle.interpolate(uvc);
    a.displace(scale * (shift + displaces.get(a.uv)));
    b.displace(scale * (shift + displaces.get(b.uv)));
    c.displace(scale * (shift + displaces.get(c.uv)));

    Triangle abc;  //то же самое, но другая структура

    bool rightOfC; //Для определения следующей ячейки
    LastChange change; // один из {iplus, jminus, kplus, iminus, jplus, kminus}
    float delta = 1.0/float(subdiv_param); //1/N
    while(true) {
        abc.set(a, b, c);
        RTIntersection res = rayTriangleIntersection(ray, abc);
        if (res.success) {
            res.intersectionPoint.n = abc.normal();
            return res;
        }

        if (make_int3(i, j, k) == end) {
            return miss;
        }

        rightOfC = rayToPoint(ray, c) == RayToPoint::RIGHT;
        if (rightOfC) {
            a   = c;
            uva = uvc;
        } else {
            b   = c;
            uvb = uvc;
        }

        // 5 = -1 mod 6
        change = LastChange((int(change) + (rightOfC ? 1 : 5)) % 6);
        switch (change) {
        case LastChange::iminus:
            --i;
            if (i < 0) return miss;
            uvc.set((j+1)*delta,(k+1)*delta);
            break;
        case LastChange::iplus:
            ++i;
            if (i >= subdiv_param) return miss;
            uvc.set(j*delta, k*delta);
            break;
        case LastChange::jminus:
            --j;
            if (j < 0) return miss;
            uvc.set(j*delta, (k+1)*delta);
            break;
        case LastChange::jplus:
            ++j;
            if (j >= subdiv_param) return miss;
            uvc.set((j+1)*delta, k*delta);
            break;
        case LastChange::kminus:
            --k;
            if (k < 0) return miss;
            uvc.set((j+1)*delta, k*delta);
            break;
        case LastChange::kplus:
            ++k;
            if(k >= subdiv_param) return miss;
            uvc.set(j*delta, (k+1)*delta);
        }
        c = triangle.interpolate(uvc);
        //Единственный запрос к текстуре в цикле
        c.displace(scale * (shift + displaces.get(c.uv)));
    }

}

//RTIntersection rayDisplacedTriIntersection1(const Ray &ray, const Triangle &triangle,
//                                           const Texture<float> &displaces,
//                                           int subdiv_param, float shift, float scale)
//{
//    Triangle abc; //микротреугольник
//    Baricentric uva, uvb, uvc; //beta-gamma для каждой вершины abc в исходном треугольнике
//    float delta = 1.0/float(subdiv_param);
//    //    float3 cNormal; //нормаль к вершине с
//    int i, j, k; //тройка индексов вершин текущей ячейки от 0 до subdiv_param
//    for (k = 0; k < subdiv_param; ++k) {
//        for (j = 0; j < subdiv_param; ++j) {
//            for (i = subdiv_param - 2 - j - k; i < subdiv_param - j - k; ++i) {
//                if (i < 0) continue;
//                if (i + j + k == subdiv_param - 1) { //Нижний микротреугольник
//                    uva.set( j   *delta,  k   *delta);
//                    uvb.set((j+1)*delta,  k   *delta);
//                    uvc.set( j   *delta, (k+1)*delta);
//                } else if (i + j + k == subdiv_param - 2) { //верхний микротреугольник
//                    uva.set((j+1)*delta,  k   *delta);
//                    uvb.set((j+1)*delta, (k+1)*delta);
//                    uvc.set(j    *delta, (k+1)*delta);
//                } else {
//                    printf ("[rayDisplacedTriIntersection]: ERROR\n");
//                }
//                abc = triangle.getMicrotriangle(uva, uvb, uvc);
//                float h0 = scale * (shift + displaces.get(abc.uv0.x, abc.uv0.y));
//                float h1 = scale * (shift + displaces.get(abc.uv1.x, abc.uv1.y));
//                float h2 = scale * (shift + displaces.get(abc.uv2.x, abc.uv2.y));
//                abc.displace(h0, h1, h2);
//                abc.setDefaultNormals(); //Пересчитать нормали как перпендикуляр к плоскости треугольника
//                RTIntersection hit = rayTriangleIntersection(ray, abc);
//                if (hit.success) {
//                    return hit;
//                }
//            }
//        }
//    }
//    RTIntersection miss;
//    miss.success = false;
//    return miss;
//}

void test(const Texture<float> &displaces)
{
    Ray ray(make_float3(0, -2, 0), make_float3(0, 1, 0), 0, INFINITY);
    Triangle T;
    T.set(make_float3(-3, 0, 0), make_float3(2, 0, 0), make_float3(-1, 2, 0));
    T.n0 = T.n1 = T.n2 = T.normal();
    T.uv0 = make_float2(0, 0);
    T.uv1 = make_float2(1, 0);
    T.uv2 = make_float2(0, 1);
    Triangle T2 = T;
    T2.displace(1, 1, 1);
    int3 start_cell, end_cell;
    Baricentric uva, uvb, uvc;
    bool res = displaceInitPhase(ray, T, T2, &start_cell, &end_cell, &uva, &uvb, &uvc, 5);
    printf("res = %d\nstart = (%d, %d, %d), end = (%d, %d, %d)\n", res, \
           start_cell.x, start_cell.y, start_cell.z,
           end_cell.x, end_cell.y, end_cell.z);
//    int k = 0.7f*5;
    printf("\n%d\n", LastChange((int(LastChange(2)) + 4) % 6) == LastChange::iminus);
}
#else

void test(const Texture<float> &displaces) {}

#endif
