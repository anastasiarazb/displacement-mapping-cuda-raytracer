#pragma once
#include <chrono>
#include <GLFW/glfw3.h>

class FPSCounter {
    std::chrono::high_resolution_clock::time_point  start, end, frame_start;
    int fps = 0;
    int framesx2 = 0;
    double delta_max = 1.0/30.0;
    double delta     = 1.0/30.0;
public:
    FPSCounter() {
        start = std::chrono::high_resolution_clock::now();
    }

    void setFrameStart() {frame_start = std::chrono::high_resolution_clock::now();}
    void setTitle(GLFWwindow* window) {
        framesx2++;

        end = std::chrono::high_resolution_clock::now();
        double diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - frame_start).count() * 1e-6;
        delta = std::max(delta, delta_max);
        delta_max = (delta_max + delta) / 2.0;
        if (diff >= 2) {
            fps = framesx2 / 2.0;
            framesx2 = 0;
            glfwSetWindowTitle(window, ("fps: " + std::to_string(fps)).c_str());
            start = std::chrono::high_resolution_clock::now();
        }
    }
    double getDelta() {return delta;}
};
