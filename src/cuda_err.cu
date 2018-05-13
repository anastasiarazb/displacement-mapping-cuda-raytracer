#include <stdio.h>
#include "cuda_err.h"

bool gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code == cudaSuccess) return false;
   else {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
      return true;
   }
}
