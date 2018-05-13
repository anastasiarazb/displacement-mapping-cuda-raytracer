#include "gpu_buffs.h"
#include "cuda_err.h"
#include "kernel_params.h"

#include <stdio.h>

void GPU_buffs::reinitFrame(uint32_t width, uint32_t height)
{
    printf("[GPU_buffs::reinitFrame]: start\n");
    pix_num = width * height;
    if (ALL_size >= pix_num) {
        printf("[GPU_buffs::reinitFrame]: ALL_size >= pix_num\n[GPU_buffs::reinitFrame]: end\n");
        clearBuffs();
        return;
    }
    destroy();
    size_t pix_buff_len = pix_num * sizeof(Pixel);
    size_t x = ((255 + pix_buff_len)/256);
    buflen = x * 256; //Длина буфера для массива пикселей, выровненная по 256 байт
    gpuErrchk(cudaMalloc(&ALL, 2*buflen + pix_num * sizeof(curandState_t)));
    res_pixels   = (Pixel *) ALL;
    accum_pixels = (Pixel *)(ALL + buflen);
    rand_buff    = (curandState_t *)(ALL + 2 * buflen);
    printf("[GPU_buffs::reinitFrame]: ALL = %p, res_pixels = %p, accum_pixels = %p, rand_buff = %p\n",\
           ALL, res_pixels, accum_pixels, rand_buff);
    clearBuffs();
    printf("[GPU_buffs::reinitFrame]: end\n");
}

void GPU_buffs::clearBuffs()
{
    //Очистка буферов пикселей res_pixels и accum_pixels
    gpuErrchk(cudaMemset(ALL, 0, 2 * buflen));
    initRandom();
//    printf("[GPU_buffs::clearBuffs]: cudaMemset(ALL, 0)\n");
}

void GPU_buffs::destroy()
{
    if (ALL != nullptr) {
        gpuErrchk(cudaFree(ALL));
        printf("[GPU_buffs::destroy]: cudaFree(ALL)\n");
    }
    ALL = nullptr;
    res_pixels   = nullptr;
    accum_pixels = nullptr;
    rand_buff    = nullptr;
}


//-----------------------------------------------
//--------------------------------------------------------

__global__
void kernel_initRandom(curandState_t *randomHelper, uint buf_size)
{
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= buf_size) {
        return;
    }

    curandState_t* state = randomHelper + id;
    curand_init(id, /* the seed controls the sequence of random values that are produced */
                  0, /* the sequence number is only important with multiple cores */
                  0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                  state);

}

void GPU_buffs::initRandom()
{
    dim3 threadsInBlock(BLOCKSIZE, 1, 1);
    dim3 numblocks(countBlocks(pix_num));
    kernel_initRandom<<<numblocks, threadsInBlock>>>(rand_buff, pix_num);
}
