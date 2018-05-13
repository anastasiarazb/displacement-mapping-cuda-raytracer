#include "shared_window.h"
#include "main.h"
#include <unistd.h>
std::atomic_bool draw_thread_alive { false };
std::atomic_bool command_to_stop { false };
std::atomic_bool work_drawed { false };
std::mutex take_framebuffer;
std::mutex thread_safe_end;
FPSCounter fps;

void draw_thread(HLBVH *bvh) {
    thread_safe_end.lock();
    cudaSetDevice(0);

    bool loop = true;
    while (!command_to_stop && loop) {
        take_framebuffer.lock();
        fps.setFrameStart();
        work_drawed = false;

        if (!draw(*bvh)) {
            printf("some shit happens!!\n");
            loop = false;
        }

        take_framebuffer.unlock();
        while (!work_drawed && !command_to_stop && loop) {
            usleep(1);
        }
    }
    draw_thread_alive = false;
    thread_safe_end.unlock();
}
