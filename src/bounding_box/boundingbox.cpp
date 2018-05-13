#include "boundingbox.h"
#include "stdio.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void BoundingBox::setPoint(float x, float y, float z)
{
    x_min = x_max = x;
    y_min = y_max = y;
    z_min = z_max = z;
    initialized = true;
}

void BoundingBox::update(float x, float y, float z)
{
    if (initialized){
        if       (x < x_min) x_min = x;
        else  if (x > x_max) x_max = x;

        if       (y < y_min) y_min = y;
        else  if (y > y_max) y_max = y;

        if       (z < z_min) z_min = z;
        else  if (z > z_max) z_max = z;

    } else setPoint(x, y, z);
}

bool intersect(const BoundingBox &A, const BoundingBox &B)
{
    return (A.x_min <= B.x_max && B.x_min <= A.x_max &&
            A.y_min <= B.y_max && B.y_min <= A.y_max &&
            A.z_min <= B.z_max && B.z_min <= A.z_max);
}

void BoundingBox::merge(const BoundingBox &A) //Объединение текущего и A
{
    if (!A.initialized) return;
    if (!initialized) {
        *this = A;
        return;
    }
    if (x_min > A.x_min) x_min = A.x_min;
    if (x_max < A.x_min) x_max = A.x_max;

    if (y_min > A.y_min) y_min = A.y_min;
    if (y_max < A.y_min) y_max = A.y_max;

    if (z_min > A.z_min) z_min = A.z_min;
    if (z_max < A.z_min) z_max = A.z_max;
}

//--------------------------------------------

float3 BoundingBox::minVals() const
{
    return make_float3(x_min, y_min, z_min);
}

float3 BoundingBox::maxVals() const
{
    return make_float3(x_max, y_max, z_max);
}

void BoundingBox::print() const
{
    if (!initialized) {
        printf("[BoundingBox::print]: not initialized\n");
        return;
    } else {
        float3 min = minVals();
        float3 max = maxVals();

        printf("left low far  (min) = {%f, %f, %f}\n"
               "right up near (max) = {%f, %f, %f}\n",
               min.x, min.y, min.z,
               max.x, max.y, max.z);
    }
}

//float3 BoundingBox::putInScope(float3 p) const //Поместить перед камерой
//{
//    if (!initialized) {
//        printf("[BoundingBox::putInScope]: not initialized\n");
//        exit(1);
//    }

//    p.x -= (x_min + x_max)/2; // (x_min, x_max) -> (- (x_max-x_min)/2, (x_max-x_min)/2)
//    p.y -= (y_min + y_max)/2; // (y_min, y_max) -> (- (y_max-y_min)/2, (y_max-y_min)/2)
//    p.z -= z_max + 1; // (z_min, z_max) -> (z_min - z_max - 1, -1);

//    return p;

//}

//BoundingBox BoundingBox::putCopyInScope() const//Поместить перед камерой
//{
//    if (!initialized) {
//        printf("[BoundingBox::putItselfInScope]: not initialized\n");
//        exit(1);
//    }

//    BoundingBox res;
//    res.initialized = true;
//    res.x_max =  (x_max - x_min)/2; // (x_min, x_max) -> (- (x_max-x_min)/2, (x_max-x_min)/2)
//    res.x_min = -(x_max - x_min)/2;
//    res.y_max =  (y_max - y_min)/2; // (y_min, y_max) -> (- (y_max-y_min)/2, (y_max-y_min)/2)
//    res.y_min = -(y_max - y_min)/2;
//    res.z_min = z_min - z_max - 1.f; // (z_min, z_max) -> (z_min - z_max - 1, -1);
//    res.z_max = 0.0f;

//    return res;
//}

glm::vec3 BoundingBox::cameraInFront() const
{
    return glm::vec3((x_max + x_min)/2.f,
                     (y_max + y_min) / 2.f,
                     z_max + fmax(x_max - x_min, y_max - y_min) + 1.5f);
}

glm::vec3 BoundingBox::lightAbove() const
{
    return glm::vec3((x_max + x_min)/2.f, y_max + (y_max - y_min)/3.f, z_max + (z_max + z_min)/2.f);
}
//BoundingBox & BoundingBox:: operator= (const BoundingBox &A)
//{
//    x_min = A.x_min;
//    x_max = A.x_max;

//    y_min = A.y_min;
//    y_max = A.y_max;

//    z_min = A.z_min;
//    z_max = A.z_max;
//    return *this;
//}

#ifdef  GLM_VEC3
void BoundingBox::computeFrom(std::vector<glm::vec4> points)
{
    initialized = false;
    for (const glm::vec4 P :points)
    {
        update(P.x, P.y, P.z);
    }
}

#endif
