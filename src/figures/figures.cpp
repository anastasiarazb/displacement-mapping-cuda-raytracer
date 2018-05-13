#include "figures.h"
#include "../cuda_numerics/float4.h"

#define AMBIENT 1.0f;

void initLights(Light lights[])
{
    lights[0].center = make_float3(0, 2, -1);
    lights[0].Radius = 0.0f;
    lights[0].color = make_float4(1.0f, 0.7f, 0.7f);
    lights[0].intensity = 1.f;

    lights[1].center = make_float3(-0.3, 3, -2);
    lights[1].Radius = 0.0f;
    lights[1].color = make_float4(0.6, 1.f, 1.f);
    lights[1].intensity = 1.f;
}

void initLights(Light *&lights, unsigned int &num_of_lights, BoundingBox bb)
{
    num_of_lights = 2;
    if (lights) {
        free(lights);
    }
    lights = (Light *)malloc(2 * sizeof(Light));

    glm::vec3 pos = bb.lightAbove();
    glm::vec3 cam = bb.cameraInFront();

    lights[1].center = make_float3(pos.x, pos.y, pos.z);
    lights[1].Radius = fmax(bb.x_max-bb.x_min, bb.y_max - bb.y_min) / 50.0f;
    lights[1].color = make_float4(0.6, 0.6, 1.f);
    lights[1].intensity = 1.f;

    lights[0].center = make_float3(pos.x + 0.4f, pos.y - 0.3f, (pos.z + cam.z)/2);
    lights[0].Radius = fmin(bb.x_max-bb.x_min, bb.y_max - bb.y_min) / 50.0f;
    lights[0].color = make_float4(1.0, 1.0, 1.0);//(0.6, 1.f, 1.f);
    lights[0].intensity = 1.f;
}
