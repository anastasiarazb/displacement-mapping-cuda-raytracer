#define GLFW_INCLUDE_GLU
#include "main.h"
#include "shared_window.h"
#include "kernel_params.h"
#define LIGHTS_NUM 1

#include "test.h"

Framebuffer buffer;
Scene       scene;
Light      *lights;
unsigned int  lights_num;
Triangle   *triangles;
Material    material;
Camera      camera;

uint blocks_per_frame = 200;//(600*800 + BLOCKSIZE - 1)/BLOCKSIZE; //800
double cursorX, cursorY, delta = 1.0 / 60.0;

bool need_refresh_scene = false;
bool need_cuda_clear_buffs = false;

const char *modelpath = "../models/triangle.obj"; float scale = 0.1f;
//const char *displpath = "../models/10.jpg"; float shift = 0.0f;
const char *displpath = "../models/chesterfield-height.jpg"; float shift = 0.0f;
//const char *displpath = "../models/iu9.png"; float shift = 0.0f;
//const char *displpath = "../models/iu9low.png"; float shift = -1.0f;
//const char *displpath = "../models/iu9-bold-low.png"; float shift = -1.0f;
//monsterfrog
//const char *modelpath = "../models/monsterfrog/monsterfrog_mapped.obj";
//const char *displpath = "../models/monsterfrog/monsterfrog-d.bmp"; float shift = -0.5; float scale = 1.f;
//const char *texturepath = "../models/green.bmp";
const char *texturepath = "../models/blue.png";
//const char *texturepath = "../models/bumped/monsterfrog/monsterfrog-n.bmp";

//cube
//const char *displpath = "../models/black.bmp"; float shift = 0.0f;// float scale = 0.6f;
//const char *modelpath = "../models/cube.obj"; float scale = 0.03f; //float scale = 0.03f;
//const char *texturepath = "../models/green.bmp";
//const char *texturepath = "../models/testBMP2x2.bmp";

int subdiv_parameter = SUBDIV_PARAMETER;

extern void test(const Texture<float> &displaces);

uint blocks_offset = 0;
bool draw(HLBVH &bvh)
{
    return cuda_main(bvh, buffer.canvas, buffer.width*buffer.height, blocks_offset, subdiv_parameter, scale);
}

void refreshScene()
{
    scene.setCamera(camera);
    scene.update();
}

GLFWwindow* initOpenGL()
{
    if(!glfwInit())
    {
        printf("glfwInit failed\n");
        exit(-1);
    }

    glfwSetErrorCallback(error_callback);

    GLFWwindow* window;
    window = glfwCreateWindow(800, 600, "CUDA & OpenGL", NULL, NULL);
    if (window == NULL)
    {
        printf("glfwOpenWindow failed.\n");
        glfwTerminate();
        exit(-2);
    }

    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetFramebufferSizeCallback(window, resize_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwGetCursorPos(window, &cursorX, &cursorY);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetWindowTitle(window, "Displacement mapping\n");
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    resize_callback(window, 800, 600);
    return window;
}

const char *usage = \
        "Usage: <program_name> <path_to_model(.obj)> <path_to_color_texture(.jpg .png .gif .bmp)>"\
        "<path_to_displacement_map(.jpg .png .gif .bmp)> <zero_value(float point)> <displace_amount(float point)>"\
        " <subdivision_parameter(integer)";

int main(int argc, char* argv[])
{
    std::string path_to_model(modelpath);
    std::string path_to_color_texture(texturepath);
    std::string path_to_displacement_map(displpath);
try{
    if (argc > 1) path_to_model = argv[1];
    if (argc > 2) path_to_color_texture = argv[2];
    if (argc > 3) path_to_displacement_map   = argv[3];
    if (argc > 4) shift = -std::stof(argv[4]);
    if (argc > 5) scale = std::stof(argv[5]);
    if (argc > 6) subdiv_parameter = std::stoi(argv[6]);
} catch (...) {
        printf("[main]: ERROR: incorrect arguments.\n%s\nExit.\n", usage);
        return 1;
}
    if (argc > 7) {
        printf("[main]: too many arguments. Should be 6, but %d are passed. Last values are ignored\n%s\n", argc-1, usage);
    } else if (argc < 7) {
        printf("[main]: there are 6 arguments to pass, %d are passed. Undefined values are set to default.\n%s\n", \
               argc-1, usage);
    } else if (argc == 0) printf("[main]: no arguments are passed, default values are used\n");
    printf("[main]: values:\n path to model = %s\n path to color texture = %s\n path to displacement map = %s\n"\
           " zero value = %f\n displacement amount = %f\n subdivision_parameter = %d\nPress Esc to exit.\n",\
           path_to_model.c_str(), path_to_color_texture.c_str(), path_to_displacement_map.c_str(), \
           0.0f-shift, scale, subdiv_parameter);

    cudaSetDevice(0);
    GpuStackAllocator::getInstance().resize(512 * 1024 * 1024);

    Texture<float4> texture;
    Texture<float> displace;
    texture.initTexture();
    displace.initTexture();
    bool resT = texture.load(path_to_color_texture.c_str());
    bool resD = displace.load(path_to_displacement_map.c_str());
    if (resT && resD) {
        displace.colorTransform(shift, scale);
        printf("[main]: Textures loaded\n");
    } else {
        printf("[main]: Textures loading failed\n");
        texture.destroy();
        displace.destroy();
        exit(1);
    }
#ifdef TEST
    test(displace);
    texture.destroy();
    displace.destroy();
    return 0;
#endif

    Model model = new_model();
    printf("[main]: Importing model\n");
    bool  res = readObjFile(path_to_model.c_str(), model, lights, lights_num);
//    bool  res = readObjFileD(modelpath, model, lights, lights_num, displace, 3, 0.0, 0.0);
    triangles = model.triangles;
//    for (size_t i = 0; i < model.num_of_triangles; ++i) {\
        triangles[i].print();\
    }

//    return 0;
    size_t triangles_num = model.num_of_triangles;

    camera.Pos = model.bb.cameraInFront();
    scene.initScene();
    scene.setCamera(camera);

    printf("[main]: Importing model %s. Imported %u triangles\n",  \
           res ? "SUCCESS" : "FAILED", model.num_of_triangles);

    if (!res) {
        model.destroy();
        texture.destroy();
        displace.destroy();
        exit(1);
    }

//    initLights(lights, lights_num, model.bb);
    material.initMaterial(2);

    printf("[main]: Loading model to GPU\n");
    cuda_load_to_gpu(triangles, triangles_num, lights, LIGHTS_NUM, material, texture, displace);

    printf("[main]: Computing HLBVH\n");
    AABB *aabb = GpuStackAllocator::getInstance().alloc<AABB>(triangles_num);
    build_aabb(aabb, triangles_num, scale);
    HLBVH bvh;
    bvh.build(aabb, triangles_num);
    GpuStackAllocator::getInstance().free(aabb);
//    return 0;

    printf("[main]: Open window\n");
    GLFWwindow* window = initOpenGL();

    printf("[main]: Start rendering loop\n");

    FPSCounter fps_counter;

    fps_counter.setFrameStart();

    draw_thread_alive = true;
    std::thread draw_th(draw_thread, &bvh);
    draw_th.detach();

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window) && draw_thread_alive)
    {
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e-3;
        if (delta >= 1.0 / 60.0) {
            if (take_framebuffer.try_lock() && draw_thread_alive) {
                if (need_refresh_scene) {
                    refreshScene();
                    need_refresh_scene = false;
                }
                if (need_cuda_clear_buffs) {
                    cuda_clear_buffs();
                    blocks_offset = 0;
                    need_cuda_clear_buffs = false;
                }

                buffer.loadBuf();
                fps.setTitle(window);
                glfwSwapBuffers(window);

                work_drawed = true;
                take_framebuffer.unlock();
            }
            glfwPollEvents();
            start = std::chrono::high_resolution_clock::now();
        }
    }
    command_to_stop = true;
    thread_safe_end.lock();

    glfwDestroyWindow(window);
    // clean up and exit
    glfwTerminate();

    cuda_free_context();
    material.destroy();
    texture.destroy();
    displace.destroy();
    free(lights);
    printf("[main]: free(lights)\n");
    model.destroy();

    return 0;
}
