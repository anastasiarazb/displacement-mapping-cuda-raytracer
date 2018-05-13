#pragma once

#include "../figures/figures.h"
#include "../bounding_box/boundingbox.h"

struct Model {
    Triangle *triangles;
    float4   *texture;
    u_int32_t num_of_triangles;
    u_int32_t num_of_texels;
    BoundingBox bb;

    void initModel();
    Model() {}
    void destroy();
    static void destroyGPU(Model *gpu_model);

    void loadToGPU(Model *gpu_model);
};

Model new_model();
