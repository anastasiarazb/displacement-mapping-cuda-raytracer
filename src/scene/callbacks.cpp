#include "callbacks.h"
#include "../cuda_app.h"
#include "framebuffer.h"
#include "../figures/figures.h"
#include "../cuda_numerics/float3.h"
#include <tuple>
#include <stdio.h>
#include "../camera/camera.h"
#include "main.h"

extern void refreshScene();

extern Framebuffer buffer;
extern __constant__ struct Scene gpu_context;
extern Light lights[];
extern Camera camera;
extern double cursorX, cursorY, delta;

void cursor_callback(GLFWwindow* window, double x, double y)
{
   camera.rotateMouse(x - cursorX, y - cursorY, delta);
   cursorX = x;
   cursorY = y;
   need_refresh_scene = true;
   need_cuda_clear_buffs = true;
   std::ignore = window;
}

void mouse_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(action != GLFW_PRESS) return;

    if(button == GLFW_MOUSE_BUTTON_LEFT)
    {

    } else
    if(button == GLFW_MOUSE_BUTTON_RIGHT)
    {

    }
    std::ignore = window;
    std::ignore = mods;
}

void resize_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);

    buffer.reinitBuffer(width, height);
    cuda_reinit_frame(width, height);
    std::ignore = window;

}

void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch(key) //GLFW_KEY_ESCAPE GLFW_KEY_LEFT GLFW_KEY_RIGHT GLFW_KEY_SPACE
        {
        case (GLFW_KEY_ESCAPE):
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case(GLFW_KEY_A):
            camera.shiftLeft(delta);
            break;
        case(GLFW_KEY_S):
            camera.shiftBackwards(delta);
            break;
        case(GLFW_KEY_D):
            camera.shiftRight(delta);
            break;
        case(GLFW_KEY_W):
            camera.shiftForward(delta);
            break;
        case(GLFW_KEY_N):
            camera.rotateUp();
            break;
        case(GLFW_KEY_M):
            camera.rotateDown();
            break;
        }
//        std::cout << camera << std::endl;
        need_refresh_scene = true;
        need_cuda_clear_buffs = true;
//        cuda_update_scene();
    }
    std::ignore = window;
    std::ignore = scancode;
    std::ignore = mods;
}

void error_callback(int error, const char* description)
{
    fputs(description, stderr);
    std::ignore = error;
}
