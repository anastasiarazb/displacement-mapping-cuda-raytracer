#ifndef RASTERIZATION_H
#define RASTERIZATION_H

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <list>
#include <math.h>
#include "../pixel.h"

struct Framebuffer {
    uint32_t width;
    uint32_t height; //Начало системы координат - в левом нижнем углу

    uint64_t size = 0;
    Pixel *canvas;

    Framebuffer();
    Framebuffer(uint64_t width, uint64_t height);
    ~Framebuffer();
    void reinitBuffer(uint32_t width, uint32_t height);
    void clearCanvas();

    void loadBuf();
    Pixel& access(uint32_t x, uint32_t y);
    void drawPoint(uint32_t x, uint32_t y);
    void drawPoint(uint32_t x, uint32_t y, float *v3color);
    bool inBuffer(uint32_t x, uint32_t y) const;
};

std::ostream& operator<<(std::ostream& os, const Framebuffer& F);

#endif // RASTERIZATION_H
