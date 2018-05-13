#pragma once
#include <cuda_runtime_api.h>

#define gpuErrchk(ans)  gpuAssert((ans), __FILE__, __LINE__, false)

bool gpuAssert(cudaError_t code, const char *file, int line, bool abort);
