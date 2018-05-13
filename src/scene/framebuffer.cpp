#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "framebuffer.h"

float red[4] = {1.f, 0,   0,   1.f};
float green[4] = {0.1f,  1.f, 0.1f,  1.f};
float blue[4] = {0,   0,   1.f, 1.f};
float white[4] = {1.f, 1.f, 1.f, 1.f};
float black[4] = {0,    0,   0,  1.f};

#define BACKGR_BRIGHTNESS 100
#define FOREGROUND_COLOR red

const uint8_t background[4] = {BACKGR_BRIGHTNESS, BACKGR_BRIGHTNESS, BACKGR_BRIGHTNESS, 255};

Framebuffer::Framebuffer()
{
    size = 0;
    canvas = nullptr;
}

Framebuffer::Framebuffer(uint64_t width, uint64_t height)
{
    this->width = width;
    this->height = height;
    size = width*height;
    canvas = (Pixel *)calloc(sizeof(Pixel), size);
    memset(canvas, BACKGR_BRIGHTNESS, size * sizeof(Pixel));
}

Framebuffer::~Framebuffer()
{
    if (this->size) {
        free(this->canvas);
        size = 0;
        width = height = 0;
    }
}

void Framebuffer::reinitBuffer(uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;
    uint64_t new_size = width*height;
    if (size == 0) {
        canvas = (Pixel *)calloc(sizeof(Pixel), new_size);
        size = new_size;
    }
    else if (size < new_size) {
        Pixel *temp = (Pixel *)calloc(sizeof(Pixel), new_size);
//        printf("%p -> %p\n", canvas, temp);
        free(canvas);
        canvas = temp;
        size = new_size;
    } //Если кадр меньше исходного, буфер не трогаем

    clearCanvas();

}

void Framebuffer::clearCanvas()
{
    if (size) {
        memset(canvas, BACKGR_BRIGHTNESS, width * height * sizeof(Pixel));
    }
}

/*___________________________________*/

Pixel& Framebuffer::access(uint32_t x, uint32_t y)
{
    if (x >= width || y >= height)
    {
        printf("Framebuffer::operator[]: pixel {%d, %d} is out of range %dx%d\n", x, y, width, height);
        exit(1);
    }
    return canvas[width*y + x];
}

/* ************************************************************ */

/*COMPARATORS, COUT*/

inline bool Framebuffer::inBuffer(uint32_t x, uint32_t y) const
{
    return x < width && y < height;

}


std::ostream& operator<<(std::ostream& os, const Framebuffer& F)
{
    os << "Framebuffer:\n " << F.width << "x" << F.height <<
          ";  size = " << F.size << std::endl;
    return os;
}


/* ___________________PRINT___________________ */


void Framebuffer::loadBuf()
{
    if (size)
    {
//        clearCanvas();
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // выравнивание
//        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, canvas);
        glDrawPixels(width, height, GL_RGBA, GL_FLOAT, canvas);
        return;
    }
}


void Framebuffer::drawPoint(uint32_t x, uint32_t y)
{
    if (inBuffer(x, y)) {
        access(x, y).setv4(FOREGROUND_COLOR);
    }
}

void Framebuffer::drawPoint(uint32_t x, uint32_t y, float *v3color)
{
    if (inBuffer(x, y)) {
        access(x, y).setv3(v3color);
    }
}

/* *********************************************** */
