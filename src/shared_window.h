#include <thread>
#include <mutex>
#include "hlbvh/hlbvh.h"
#include "GLFW/glfw3.h"
#include "timer/FPScounter.h"
#include <atomic>

void draw_thread(HLBVH *bvh);
extern std::atomic_bool draw_thread_alive;
extern std::atomic_bool command_to_stop;
extern std::atomic_bool work_drawed;
extern std::mutex take_framebuffer;
extern std::mutex thread_safe_end;
extern FPSCounter fps;
