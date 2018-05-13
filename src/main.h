#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "scene/framebuffer.h"
#include "figures/figures.h"
#include "scene/callbacks.h"
#include "cuda_app.h"
#include "scene/scene.h"
#include "cuda_numerics/float3.h"
#include "textures/material.h"
#include "textures/texture.h"
#include "parsers/obj.h"
#include "model/model.h"
#include "hlbvh/hlbvh.h"
#include "timer/FPScounter.h"
#include <string>

bool draw(HLBVH &bvh);
extern bool need_refresh_scene;
extern bool need_cuda_clear_buffs;
extern uint blocks_per_frame;
