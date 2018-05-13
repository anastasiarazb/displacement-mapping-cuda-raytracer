#pragma once
#include <cuda_runtime_api.h>
//--------------------------------------------------------------

#define BLOCKSIZE 128

__host__ __device__
unsigned int countThreads(unsigned int size);


__host__ __device__
unsigned int countBlocks(unsigned int size);
