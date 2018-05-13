#pragma once

#include <cuda_runtime.h>
#include "../figures/figures.h"
#include "../camera/camera.h"

struct Scene {

    float scale;
    size_t frames_counter;

    uint32_t width, height;
    float projectionPlaneZ; //Плоскость проецирования Z = -1
    float3 eye;

    float LookAt[16];/* = glm::perspective(45, width/hight,  0.1, 100.0f);*/
    float ScreenToWorld[16];

    void initScene();
    void reinitScene(uint32_t width, uint32_t height);
    void setCamera(const Camera &camera);
//    void setEye(float x, float y, float z);

    void sendToGPU();
    void update();
};
