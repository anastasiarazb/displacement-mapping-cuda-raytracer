#include "model.h"
#include "../cuda_err.h"

void Model::loadToGPU(Model *gpu_model)
{
    Model cpu_copy = *this;
    cpu_copy.triangles = nullptr;
    gpu_model = nullptr;
    //Выделить память под массив треугольников на GPU, скопировать
    gpuErrchk(cudaMalloc(&(cpu_copy.triangles), num_of_triangles * sizeof(Triangle)));
    gpuErrchk(cudaMemcpy(cpu_copy.triangles, this->triangles,
                         num_of_triangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    //Выделить память под структуру на GPU и отправить туда ее копию
    gpuErrchk(cudaMalloc(&gpu_model, sizeof(Model)));
    gpuErrchk(cudaMemcpy(gpu_model, &cpu_copy, sizeof(Model), cudaMemcpyHostToDevice));
}

void Model::destroyGPU(Model *gpu_model)
{
    if (gpu_model != nullptr) {
        Model cpu_copy;
        gpuErrchk(cudaMemcpy(&cpu_copy, gpu_model, sizeof(Model), cudaMemcpyDeviceToHost));
        if (cpu_copy.triangles != nullptr) {
            gpuErrchk(cudaFree(cpu_copy.triangles));
        }
        gpuErrchk(cudaFree(gpu_model));
    }
}
