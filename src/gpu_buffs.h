#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include "pixel.h"
#include "figures/figures.h"
#include "stdio.h"

struct GPU_buffs{
    char  *ALL;
    Pixel *res_pixels;
    Pixel *accum_pixels;
    curandState_t *rand_buff;

    size_t ALL_size;
    size_t buflen;
    size_t pix_num;

    void destroy();
    void reinitFrame(uint32_t width, uint32_t height);
    void clearBuffs();
    void initRandom();
};
