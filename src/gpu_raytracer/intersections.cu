#include "../figures/figures.h"
#include "../cuda_numerics/int3.h"
#include "../cuda_numerics/float2.h"
#include "../cuda_numerics/float3.h"
#include "../cuda_numerics/cuda_numerics.h"

#include "../test.h"

#ifdef SINGLE_KERNEL
#define out printf
#define PRINT(X) (X).print()
#else
#define out(...)
#define PRINT(...)
#endif

#ifndef TEST

#define BAD_INDEX make_int3(-1, -1, -1)

__device__
float sin_displace(float2 uv)
{
    return fabs(sinf(5.f*uv.x*M_PI)*sinf(5.f*uv.y*M_PI))*0.1f;
}

enum class RayToPoint {
    LEFT, RIGHT, FORWARD
};

enum class IntersectionType{
    ENTER, EXIT, NONE
};

FUNC_PREF
const char * toString(RayToPoint a)
{
    switch (a) {
    case RayToPoint::FORWARD:
        return "FORWARD";
    case RayToPoint::LEFT:
        return "LEFT";
    case RayToPoint::RIGHT:
        return "RIGHT";
    default:
        return "[RayToPointToString]: ERROR: undefined type.\n";
    }
}

FUNC_PREF
const char * toString(IntersectionType a)
{
    switch (a) {
    case IntersectionType::ENTER:
        return "ENTER";
    case IntersectionType::EXIT:
        return "EXIT";
    case IntersectionType::NONE:
        return "IntersectionType::NONE";
    default:
        return "[IntersectionTypeToString]: ERROR: undefined type.\n";
    }
}

enum class Side {
    P0P1, P1P2, P2P0, ANY, TOP, BOTTOM
};

FUNC_PREF
const char * toString(Side a)
{
    switch (a) {
    case  Side::P0P1:
        return "P0P1";
    case  Side::P1P2:
        return "P1P2";
    case  Side::P2P0:
        return "P2P0";
    case  Side::ANY:
        return "Side::ANY";
    case  Side::TOP:
        return "Side::TOP";
    case  Side::BOTTOM:
        return "Side::BOTTOM";
    default:
        return "";
    }
}

enum class LastChange {
    iplus = 0, jminus = 1, kplus = 2, iminus = 3, jplus = 4, kminus = 5 //0-5
};

FUNC_PREF
const char * toString(LastChange a)
{
    switch (a) {
    case LastChange::iplus:
        return "iplus";
    case LastChange::jplus:
        return "jplus";
    case LastChange::kplus:
        return "kplus";
    case LastChange::iminus:
        return "iminus";
    case LastChange::jminus:
        return "jminus";
    case LastChange::kminus:
        return "kminus";
    default:
        return "";
    }
}

//---------------------------------------------------------------------------

FUNC_PREF inline bool isLower(int i, int j, int k, int subdiv_param)
{
    return (i + j + k) == (subdiv_param - 1);
}

FUNC_PREF inline bool isUpper(int i, int j, int k, int subdiv_param)
{
    return (i + j + k) == (subdiv_param - 2);
}

//  с    b-a
//  |\    \|     k ^      i растет по диагонали при движении сверху вниз
//  a-b    c     j >
__device__
void setUVabcByijk(Baricentric &uva, Baricentric &uvb, Baricentric &uvc,
                   int i, int j, int k, int subdiv_param){
    float delta = 1.f/float(subdiv_param);
    if (i + j + k == subdiv_param - 1) { //Нижний микротреугольник
        uva.set( j   *delta,  k   *delta);
        uvb.set((j+1)*delta,  k   *delta);
        uvc.set( j   *delta, (k+1)*delta);
        return;
    } else if (i + j + k == subdiv_param - 2) { //верхний микротреугольник
        uva.set((j+1)*delta, (k+1)*delta);
        uvb.set( j   *delta, (k+1)*delta);
        uvc.set((j+1)*delta,  k   *delta);
        return;
    } else {
        printf("[setUVabcByijk]: Error: i+j+k=%d+%d+%d != N-1 or N-2, N = %d\n",
               i, j, k, subdiv_param);
    }
}

__device__
RayToPoint rayToPoint(const Ray &ray, const InnerPoint &P)
{
    float sign = dot(ray.dir, cross(P.n, ray.org - P.p));
    return sign == 0.0f ? RayToPoint::FORWARD : (sign < 0.0f? RayToPoint::LEFT : RayToPoint::RIGHT);
}

__device__ bool rayBetweenPoints(const Ray &ray, const InnerPoint &A, const InnerPoint &B)
{
    RayToPoint rayA = rayToPoint(ray, A);
    RayToPoint rayB = rayToPoint(ray, B);
    return (rayA != RayToPoint::LEFT) && (rayB != RayToPoint::RIGHT);
}

//-------------------------------------------------------------------
__device__
int rayTopBottomAttitude(const Ray &ray, const Triangle &triangle, \
                         int subdiv_param, int3 &micro_tri_index)
{
    RTIntersection intersection = rayTriangleIntersectionPrimary(ray, triangle);
//    Baricentric trueInt = rayTriangleIntersection(ray, triangle).coords;
//    out("[rayTopBottomAttitude]: intersection uv = (%f, %f)\n", trueInt.beta(), trueInt.gamma());
    float sign = dot(ray.dir, triangle.normal());
    //    out("[rayTopBottomAttitude]: Triangle = <(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)>\n"\
    "                        normal() = (%f, %f, %f), sign = %f, intersection = %d\n", \
    triangle.p0.x,triangle.p0.y, triangle.p0.z,\
            triangle.p1.x,triangle.p1.y, triangle.p1.z, \
            triangle.p2.x,triangle.p2.y, triangle.p2.z, \
            triangle.normal().x, triangle.normal().y, triangle.normal().z, sign, intersection.success);
    if (!intersection.success || sign == 0.0f) {
        //        out("[rayTopBottomAttitude]: success = %d, sign = %f, uv = (%f, %f)\n", intersection.success, sign, \
        intersection.coords.beta(), intersection.coords.gamma());
        micro_tri_index = BAD_INDEX;
        return 0;
    } else if (sign < 0.0f) { //Нормаль и луч разнонаправлены => луч входит
        micro_tri_index.x = intersection.coords.alpha() *subdiv_param;
        micro_tri_index.y = intersection.coords.beta()  *subdiv_param;
        micro_tri_index.z = intersection.coords.gamma() *subdiv_param;
        if (micro_tri_index.x + micro_tri_index.y + micro_tri_index.z > subdiv_param - 1) {
            printf("[rayTopBottomAttitude]: i + j + k = %d + %d + %d = %d, a-beta-gamma=(%f, %f, %f)\n",\
                   micro_tri_index.x, micro_tri_index.y, micro_tri_index.z, \
                   micro_tri_index.x + micro_tri_index.y + micro_tri_index.z, \
                   intersection.coords.alpha(), intersection.coords.beta(), intersection.coords.gamma());
        }
        return -1;
    } else {
        micro_tri_index.x = intersection.coords.alpha() *subdiv_param;
        micro_tri_index.y = intersection.coords.beta()  *subdiv_param;
        micro_tri_index.z = intersection.coords.gamma() *subdiv_param;
        if (micro_tri_index.x + micro_tri_index.y + micro_tri_index.z > subdiv_param - 1) {
            printf("[rayTopBottomAttitude]: i + j + k = %d + %d + %d = %d, a-beta-gamma=(%f, %f, %f)\n",\
                   micro_tri_index.x, micro_tri_index.y, micro_tri_index.z, \
                   micro_tri_index.x + micro_tri_index.y + micro_tri_index.z, \
                   intersection.coords.alpha(), intersection.coords.beta(), intersection.coords.gamma());
        }
        return 1;
    }
    //    return IntersectionType::NONE;
}

__device__
IntersectionType findTopBottomIntersections(const Ray &ray, const Triangle &triangle, const Side side,\
                                            int3 &micro_tri_index, \
                                            Baricentric &uva,  Baricentric &uvb, Baricentric &uvc, \
                                            LastChange &change, const int subdiv_param)
{
    int sign = rayTopBottomAttitude(ray, triangle, subdiv_param, micro_tri_index);
    if (sign == 0) {
        micro_tri_index = BAD_INDEX;
        return IntersectionType::NONE;
    }
    if ((side == Side::TOP    && sign > 0) ||
            (side == Side::BOTTOM && sign < 0)) {
        return IntersectionType::EXIT;
    }
    Baricentric uvA, uvB, uvC;
    const int i = micro_tri_index.x;
    const int j = micro_tri_index.y;
    const int k = micro_tri_index.z;
    setUVabcByijk(uvA, uvB, uvC, i, j, k, subdiv_param);
    InnerPoint A = triangle.interpolate(uvA);
    InnerPoint B = triangle.interpolate(uvB);
    InnerPoint C = triangle.interpolate(uvC);
    //  C    B-A
    //  |\    \|    k ^      i растет по диагонали при движении вниз влево
    //  A-B    C    j >

    if (rayBetweenPoints(ray, A, B)) {
        uva = uvA;
        uvb = uvB;
        uvc = uvC;
        if (isLower(i, j, k, subdiv_param)) {
            change = LastChange::kplus;
        } else {
            change = LastChange::kminus;
        }
        return IntersectionType::ENTER;
    } else if (rayBetweenPoints(ray, B, C)) {
        uva = uvB;
        uvb = uvC;
        uvc = uvA;
        if (isLower(i, j, k, subdiv_param)) {
            change = LastChange::iplus;
        } else {
            change = LastChange::iminus;
        }
        return IntersectionType::ENTER;
    } else if (rayBetweenPoints(ray, C, A)) {
        uva = uvC;
        uvb = uvA;
        uvc = uvB;
        if (isLower(i, j, k, subdiv_param)) {
            change = LastChange::jplus;
        } else {
            change = LastChange::jminus;
        }
        return IntersectionType::ENTER;
    } else {
        //В надежде, что по крайней мере луч пройдет насквозь и встретится с пересечением на 1-й же итерации
        uva = uvC;
        uvb = uvA;
        uvc = uvB;
        change = LastChange::kplus;
        out("[findTopBottomIntersections]: ray is not in ABC, ijk = %d, %d, %d\n" \
            "uvA = (%f, %f), uvb = (%f, %f), uvc = (%f, %f)\n"\
            "ray-A = %s, ray-B = %s, ray-C = %s\n", i, j, k, \
            uvA.beta(), uvA.gamma(), uvB.beta(), uvB.gamma(), uvC.beta(), uvC.gamma(), \
            toString(rayToPoint(ray, A)), toString(rayToPoint(ray, B)), toString(rayToPoint(ray, C)));
        return IntersectionType::ENTER;
    }
}
__device__
bool raySideAttitude(const Ray &ray, const InnerPoint A, const InnerPoint B, const int subdiv_param, int &micro_tri_param)
{
    //Случай вхождения луча в призму через сторону AB: A-^-B
    float delta = 1.0f/float(subdiv_param);
    InnerPoint a, b;
    b = A;
    for (int i = 0; i < subdiv_param - 1; ++i) {
        a = b;
        b.setInterpolate(A, B, float(i+1)*delta);
        if (rayBetweenPoints(ray, a, b)) {
            micro_tri_param = i;
            return true;
        }
    }
    a = b;
    b = B; //ибо возможно, что N * (1/N) != 1
    if (rayBetweenPoints(ray, a, b)) {
        micro_tri_param = subdiv_param - 1;
        return true;
    }
    micro_tri_param = -1;
    return false;
}

__device__
bool findStartSide(const Ray &ray, const Triangle &triangle, const Side side, int3 &start_cell,
                   Baricentric &uva, Baricentric &uvb, Baricentric &uvc, LastChange &change, const int subdiv_param)
{
    out("[findStartSide]: i'm here\n");
    InnerPoint P0 = triangle.getVertex0();
    InnerPoint P1 = triangle.getVertex1();
    InnerPoint P2 = triangle.getVertex2();
    int cell_index = -1;
    start_cell = BAD_INDEX;
    int i, j, k;
    switch (side) {
    case Side::P0P1:
        if (raySideAttitude(ray, P0, P1, subdiv_param, cell_index)) {
            start_cell.x = i = subdiv_param - cell_index - 1;
            start_cell.y = j = cell_index;
            start_cell.z = k = 0;
            change = LastChange::kplus;
            setUVabcByijk(uva, uvb, uvc, i, j, k, subdiv_param);
            return true;
        } else {
            return false;
        }
    case Side::P1P2:
        if (raySideAttitude(ray, P1, P2, subdiv_param, cell_index)) {
            start_cell.x = i = 0;
            start_cell.y = j = subdiv_param - cell_index - 1;
            start_cell.z = k = cell_index;
            change = LastChange::iplus;
            setUVabcByijk(uvc, uva, uvb, i, j, k, subdiv_param);
            return true;
        } else {
            return false;
        }
    case Side::P2P0:
        if (raySideAttitude(ray, P2, P0, subdiv_param, cell_index)) {
            start_cell.x = i = cell_index;
            start_cell.y = j = 0;
            start_cell.z = k = subdiv_param - cell_index - 1;
            change = LastChange::jplus;
            setUVabcByijk(uvb, uvc, uva, i, j, k, subdiv_param);
            return true;
        } else {
            return false;
        }
    default:
        printf("[findStartSide]: no action for %s is introduced\n", toString(side));
        return false;
    }
}

//--------------------------------------------

__device__
RTIntersection traverse(const Ray &ray, const Triangle &triangle, const Texture<float> &displaces, \
                        int3 start_cell, Baricentric uva, Baricentric uvb, Baricentric uvc, LastChange change,
                        int subdiv_param, uint thread_idx)
{
    int i = start_cell.x;
    int j = start_cell.y;
    int k = start_cell.z;
    InnerPoint a, b, c; //текущий микротреугольник
    Triangle abc;  //то же самое, но другая структура
    a = triangle.interpolate(uva);
    b = triangle.interpolate(uvb);
    c = triangle.interpolate(uvc);
#ifdef PROCEDURE_DISPLACE
    a.displace(sin_displace(a.uv));
    b.displace(sin_displace(b.uv));
    c.displace(sin_displace(c.uv));
#else
    a.displace(displaces.get(a.uv));
    b.displace(displaces.get(b.uv));
    c.displace(displaces.get(c.uv));
#endif
    out("[traverse]: i'm here: a.uv = (%f, %f), b.uv = (%f, %f), c.uv = (%f, %f)\n", a.uv.x, a.uv.y, b.uv.x, b.uv.y, c.uv.x, c.uv.y);
    bool rightOfC; //Для определения следующей ячейки
    float delta = 1.0f/float(subdiv_param); //1/N
    int max_steps = subdiv_param * 2; //Хватило бы и subdiv_param*2 - 1, но так вернее
    RTIntersection miss; miss.success = false;
    while(--max_steps) {
//        if (!rayBetweenPoints(ray, a, b)){
//            out("[rayDisplacedTriIntersection]: !rayBetweenPoints(ray, a, b): rayA = %s, rayB = %s, \n" \
//                "uva = (%f, %f), uvb = (%f, %f), ijk = %d, %d, %d\n", \
//                toString(rayToPoint(ray, a)), toString(rayToPoint(ray, b)),\
//                uva.beta(), uva.gamma(), uvb.beta(), uvb.gamma(), i, j, k);
//        }
        abc.set(a, b, c);
        out("[rayDisplacedTriIntersection]: ijk = (%d, %d, %d),\n"
            "a   = (%f, %f, %f), b   = (%f, %f, %f), c   = (%f, %f, %f), rightOfC = %d, %s\n"
            "uva = (%f, %f, %f), uvb = (%f, %f, %f), uvc = (%f, %f, %f)\n\n", \
            i, j, k, COORDS(a.p), COORDS(b.p), COORDS(c.p), rightOfC, toString(rayToPoint(ray, c)),
            uva.alpha(), uva.beta(), uva.gamma(), \
            uvb.alpha(), uvb.beta(), uvb.gamma(), \
            uvc.alpha(), uvc.beta(), uvc.gamma());
        RTIntersection res = rayTriangleIntersection(ray, abc);
        if (res.success) {
            res.intersectionPoint.n = abc.normal();
            out("[rayDisplacedTriIntersection]: intersection\n***\n");
            return res;
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
        out("[rayDisplacedTriIntersection]: last change = %s\n", toString(change));
        change = LastChange((int(change) + (rightOfC ? 1 : 5)) % 6);
        out("[rayDisplacedTriIntersection]: change = %s\n", toString(change));
        switch (change) {
        case LastChange::iminus:
            --i;
            if (i < 0) return miss;
            uvc.set(float(j+1)*delta,float(k+1)*delta);
            break;
        case LastChange::iplus:
            ++i;
            if (i >= subdiv_param) return miss;
            uvc.set(float(j)*delta, float(k)*delta);
            break;
        case LastChange::jminus:
            --j;
            if (j < 0) return miss;
            uvc.set(float(j)*delta, float(k+1)*delta);
            break;
        case LastChange::jplus:
            ++j;
            if (j >= subdiv_param) return miss;
            uvc.set(float(j+1)*delta, float(k)*delta);
            break;
        case LastChange::kminus:
            --k;
            if (k < 0) return miss;
            uvc.set(float(j+1)*delta, float(k)*delta);
            break;
        case LastChange::kplus:
            ++k;
            if(k >= subdiv_param) return miss;
            uvc.set(float(j)*delta, float(k+1)*delta);
        }

        c = triangle.interpolate(uvc);
        //Единственный запрос к текстуре в цикле
#ifdef PROCEDURE_DISPLACE
        c.displace(sin_displace(c.uv));
#else
        c.displace(displaces.get(c.uv));
#endif
    }
    out("[traverse]: cycled %u\n", thread_idx);
    return miss;
}

/* Поиск пересечения с треугольником с наложением карты смещения
 * Возвращаемое значение: структура, содержащая координаты точки пересечения и ее атрибуты
 * Аргументы:
 * const Ray &ray - луч
 * const Triangle &triangle - треугольник, с которым ищется пересечение
 * const Texture<float> &displaces - карта смещений
 * int subdiv_param - пераметр разбиения вдоль стороны треугольника
 * float max_height - максимальная величина смещения
 * uint thread_idx  - номер работающего потока (для отладочного вывода)
 */
__device__
RTIntersection rayDisplacedTriIntersection(const Ray &ray, const Triangle &triangle,
                                           const Texture<float> &displaces,
                                           int subdiv_param, float max_height, uint thread_idx)
{
    out("*****************************\n[rayDisplacedTriIntersection]:");
    PRINT(ray);
    PRINT(triangle);
    //Торцевые крышки
    Triangle top = triangle;
    top.displace(max_height, max_height, max_height);
    Triangle bottom = triangle;
    bottom.displace(-max_height, -max_height, -max_height);

    int3        start = BAD_INDEX; //(-1, -1, -1)
    //Координаты alpha-beta-gamma для каждой вершины
    Baricentric uva, uvb, uvc;
    LastChange  change; // один из {iplus, jminus, kplus, iminus, jplus, kminus}
    RTIntersection res; res.success = false;
    if (findStartSide(ray, triangle, Side::P0P1, start, uva, uvb, uvc,
                      change, subdiv_param))
    {
        res = traverse(ray, triangle, displaces, start, uva, uvb, uvc,
                       change, subdiv_param, thread_idx);
        if (res.success) return res;
    }
    if (findStartSide(ray, triangle, Side::P1P2, start, uva, uvb, uvc, change, subdiv_param)){
        res = traverse(ray, triangle, displaces, start, uva, uvb, uvc, change, subdiv_param, thread_idx);
        if (res.success) return res;
    }
    if (findStartSide(ray, triangle, Side::P2P0, start, uva, uvb, uvc, change, subdiv_param)){
        res = traverse(ray, triangle, displaces, start, uva, uvb, uvc, change, subdiv_param, thread_idx);
        if (res.success) return res;
    }
    IntersectionType intersect_type;
    intersect_type = findTopBottomIntersections(ray, top, Side::TOP, start, uva, uvb, uvc, change, subdiv_param);
    if (intersect_type == IntersectionType::ENTER) {
        res = traverse(ray, triangle, displaces, start, uva, uvb, uvc, change, subdiv_param, thread_idx);
        if (res.success) return res;
    }
    intersect_type = findTopBottomIntersections(ray, bottom, Side::BOTTOM, start, uva, uvb, uvc, change, subdiv_param);
    if (intersect_type == IntersectionType::ENTER) {
        res = traverse(ray, triangle, displaces, start, uva, uvb, uvc, change, subdiv_param, thread_idx);
        if (res.success) return res;
    }
    out("[rayDisplacedTriIntersection]: return miss\n");
    return res;
}

#endif

__device__
RTIntersection rayDisplacedTriIntersectionBruteforce(const Ray &ray, const Triangle &triangle,
                                            const Texture<float> &displaces,
                                            int subdiv_param)
{
    Triangle abc; //микротреугольник
    Baricentric uva, uvb, uvc; //beta-gamma для каждой вершины abc в исходном треугольнике
    float delta = 1.0/float(subdiv_param);
    //    float3 cNormal; //нормаль к вершине с
    int i, j, k; //тройка индексов вершин текущей ячейки от 0 до subdiv_param
    for (k = 0; k < subdiv_param; ++k) {
        for (j = 0; j < subdiv_param; ++j) {
            for (i = subdiv_param - 2 - j - k; i < subdiv_param - j - k; ++i) {
                if (i < 0) continue;
                if (i + j + k == subdiv_param - 1) { //Нижний микротреугольник
                    uva.set( j   *delta,  k   *delta);
                    uvb.set((j+1)*delta,  k   *delta);
                    uvc.set( j   *delta, (k+1)*delta);
                } else if (i + j + k == subdiv_param - 2) { //верхний микротреугольник
                    uva.set((j+1)*delta,  k   *delta);
                    uvb.set((j+1)*delta, (k+1)*delta);
                    uvc.set(j    *delta, (k+1)*delta);
                } else {
                    printf ("[rayDisplacedTriIntersection]: ERROR\n");
                }
                abc = triangle.getMicrotriangle(uva, uvb, uvc);
                float h0 = displaces.get(abc.uv0.x, abc.uv0.y);
                float h1 = displaces.get(abc.uv1.x, abc.uv1.y);
                float h2 = displaces.get(abc.uv2.x, abc.uv2.y);
                abc.displace(h0, h1, h2);
                abc.setDefaultNormals(); //Пересчитать нормали как перпендикуляр к плоскости треугольника
                RTIntersection hit = rayTriangleIntersection(ray, abc);
                if (hit.success) {
                    return hit;
                }
            }
        }
    }
    RTIntersection miss;
    miss.success = false;
    return miss;
}


//------------------------------------------------------------------------------------
#define EPS 1e-5f

FUNC_PREF
RTIntersection rayTriangleIntersection(const Ray &ray, const Triangle &triangle)
{
    RTIntersection false_res;
    false_res.success = false;

    const float3 p0 = triangle.p0;
    const float3 p1 = triangle.p1;
    const float3 p2 = triangle.p2;

    const float A = p0.x - p1.x;
    const float B = p0.y - p1.y;
    const float C = p0.z - p1.z;

    const float D = p0.x - p2.x;
    const float E = p0.y - p2.y;
    const float F = p0.z - p2.z;

    const float G = ray.dir.x;
    const float H = ray.dir.y;
    const float I = ray.dir.z;

    const float J = p0.x - ray.org.x;
    const float K = p0.y - ray.org.y;
    const float L = p0.z - ray.org.z;

    const float EIHF = E*I-H*F;
    const float GFDI = G*F-D*I;
    const float DHEG = D*H-E*G;
    const float denom = (A*EIHF + B*GFDI + C*DHEG) ;

    float beta  = (J*EIHF + K*GFDI + L*DHEG) / denom;

    if (beta < 0.0f - EPS || beta > 1.0f + EPS) {
        false_res.coords.set(beta, -1.0f);
        return false_res;
    }
    const float AKJB = A*K - J*B;
    const float JCAL = J*C - A*L;
    const float BLKC = B*L - K*C;
    float gamma = (I*AKJB + H*JCAL + G*BLKC)/denom;
    if (gamma < 0.0f - EPS || beta + gamma > 1.0f + EPS) {
        false_res.coords.set(beta, gamma);
        return false_res;
    }
//        gamma = clamp(gamma, 0.0f, 1.0f - beta);
//        beta = clamp(beta, 0.0f, 1.0f);

    float tval = -(F*AKJB + E*JCAL + D*BLKC) / denom;

    RTIntersection res;
    res.success = tval >= ray.tmin && tval <= ray.tmax;
    res.t       = tval;
    res.intersectionPoint = triangle.interpolate(beta, gamma);
    res.coords.set(beta, gamma);
    return res;

}

FUNC_PREF
RTIntersection rayTriangleIntersectionPrimary(const Ray &ray, const Triangle &triangle)
{
    RTIntersection false_res;
    false_res.success = false;

    const float3 p0 = triangle.p0;
    const float3 p1 = triangle.p1;
    const float3 p2 = triangle.p2;

    const float A = p0.x - p1.x;
    const float B = p0.y - p1.y;
    const float C = p0.z - p1.z;

    const float D = p0.x - p2.x;
    const float E = p0.y - p2.y;
    const float F = p0.z - p2.z;

    const float G = ray.dir.x;
    const float H = ray.dir.y;
    const float I = ray.dir.z;

    const float J = p0.x - ray.org.x;
    const float K = p0.y - ray.org.y;
    const float L = p0.z - ray.org.z;

    const float EIHF = E*I-H*F;
    const float GFDI = G*F-D*I;
    const float DHEG = D*H-E*G;
    const float denom = (A*EIHF + B*GFDI + C*DHEG) ;

    float beta  = (J*EIHF + K*GFDI + L*DHEG) / denom;

    if (beta < 0.0f - EPS || beta > 1.0f + EPS) {
        false_res.coords.set(beta, -1.0f);
        return false_res;
    }

    beta = clamp(beta, 0.0f + EPS, 1.0f - EPS);
    const float AKJB = A*K - J*B;
    const float JCAL = J*C - A*L;
    const float BLKC = B*L - K*C;
    float gamma = (I*AKJB + H*JCAL + G*BLKC)/denom;
    if (gamma < 0.0f - EPS || beta + gamma > 1.0f + EPS) {
        false_res.coords.set(beta, gamma);
        return false_res;
    }
    gamma = clamp(gamma, 0.0f + EPS, 1.0f - beta - EPS);

    float tval = -(F*AKJB + E*JCAL + D*BLKC) / denom;

    RTIntersection res;
    res.success = tval >= ray.tmin && tval <= ray.tmax;
    res.t       = tval;
    res.intersectionPoint = triangle.interpolate(beta, gamma);
    res.coords.set(beta, gamma);
    return res;

}

bool __device__ rayAABBIntersection(float3 start, float3 dir, float3 aabbmin, float3 aabbmax) {
    // r.dir is unit direction vector of ray
    float3 dirfrac;
    dirfrac.x = 1.0f / dir.x;
    dirfrac.y = 1.0f / dir.y;
    dirfrac.z = 1.0f / dir.z;
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (aabbmin.x - start.x) * dirfrac.x;
    float t2 = (aabbmax.x - start.x) * dirfrac.x;
    float t3 = (aabbmin.y - start.y) * dirfrac.y;
    float t4 = (aabbmax.y - start.y) * dirfrac.y;
    float t5 = (aabbmin.z - start.z) * dirfrac.z;
    float t6 = (aabbmax.z - start.z) * dirfrac.z;

    float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
    float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
    if (tmax < 0)
    {
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        return false;
    }
    return true;
}
