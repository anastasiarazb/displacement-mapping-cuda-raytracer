#include "scene.h"
#include "glm/gtx/transform.hpp"
#include "stdio.h"
#include <glm/gtc/type_ptr.hpp>
#include "../glm_output/glm_outp.h"

void Scene::initScene()
{
    width = 800;
    height = 600;
    frames_counter = 1;
    projectionPlaneZ = -1;
    eye = make_float3(0, 0, 0);
    scale = 1e-7;
    glm::mat4 P = glm::perspective(45.0f, float(width) / float(height),
                                   -projectionPlaneZ, -1000.f*projectionPlaneZ);
    glm::mat4 InvP= glm::inverse(P);

    matrCopy(LookAt, glm::mat4(1.0));
    matrCopy(ScreenToWorld, InvP);
    printf("[initScene]: done\n");
}

void Scene::reinitScene(uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;
    frames_counter = 1;
    glm::mat4 P = glm::perspective(45.0f, float(width) / float(height),
                                   -projectionPlaneZ, -1000.f*projectionPlaneZ);
    glm::mat4 InvP = glm::inverse(P);
    glm::mat4 Look = glm::make_mat4(LookAt);
    matrCopy(ScreenToWorld, Look * InvP);
}

//void Scene::setEye(float x, float y, float z)
//{
//    eye = make_float3(x, y, z);
//}

void Scene::update()
{
    frames_counter = 1;
    sendToGPU();
}

void Scene::setCamera(const Camera &camera)
{
    frames_counter = 1; //В надежде, что буфер будет очищен
    eye = tofloat3(camera.Pos);
    projectionPlaneZ = -1; //Проекция идет на плоскость Z = -1;
    //Внутри будет производиться преобразование отражения по оси z
    glm::mat4 P = glm::perspective(45.0f, float(width) / float(height),
                                   -projectionPlaneZ, -1000.f*projectionPlaneZ);
    glm::mat4 look = camera.lookAt();
    matrCopy(LookAt, look);
    matrCopy(ScreenToWorld, look * glm::inverse(P));
//    matrCopy(ScreenToWorld, glm::inverse(P) * look);
}
