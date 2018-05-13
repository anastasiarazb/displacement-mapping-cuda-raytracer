#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include <vector>
#include <cuda_runtime.h>

#define GLM_VEC3

#ifdef  GLM_VEC3
#include <glm/glm.hpp>
#endif

struct BoundingBox
{
    bool initialized;
    float x_min, x_max,
          y_min, y_max,
          z_min, z_max;

    BoundingBox() {}

    void setPoint(float x, float y, float z);
    void update(float x, float y, float z);
    void update(float3 p) {update(p.x, p.y, p.z);}

    void merge(const BoundingBox &A); //Объединение текущего и A

//    BoundingBox & operator= (const BoundingBox &A);

    void initBoundingBox() {initialized = false;}
    void initBoundingBox(float x, float y, float z)
    {   setPoint(x, y, z);}
    void initBoundingBox(float3 p)
    {   setPoint(p.x, p.y, p.z);}

    void print() const;


    float3 minVals() const;
    float3 maxVals() const;

//    float3 putInScope(float3 p) const; //Поместить перед камерой
//    BoundingBox putCopyInScope() const; //Поместить перед камерой

    glm::vec3 cameraInFront() const;
    glm::vec3 lightAbove()    const;

#ifdef  GLM_VEC3
    inline void setPoint(glm::vec3 point){setPoint(point.x, point.y, point.z);}
    inline void update(glm::vec3 point) {update(point.x, point.y, point.z);}
    void computeFrom(std::vector<glm::vec4> points);
    BoundingBox(glm::vec3 point) {setPoint(point.x, point.y, point.z);}
#endif
};
bool intersect(const BoundingBox &A, const BoundingBox &B);
#endif // BOUNDINGBOX_H
