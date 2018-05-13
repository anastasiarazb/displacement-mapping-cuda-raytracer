#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../figures/figures.h"
#include "../bounding_box/boundingbox.h"
#include "../model/model.h"


bool readObjFile(const char *path,
                 Model  &model,
                 Light *&lights, unsigned int &num_of_lights);

bool readObjFileD(const char *path,
                  Model  &model,
                  Light *&lights, unsigned int &num_of_lights,
                  const Texture<float> &displaces, int subdiv_param, float shift = 0.0, float scale = 1.0f);
