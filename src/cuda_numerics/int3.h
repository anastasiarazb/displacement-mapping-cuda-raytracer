#pragma once

#include "cuda_numerics.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

static FUNC_PREFIX
bool operator == (int3 a, int3 b)
{
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

static FUNC_PREFIX
bool operator != (int3 a, int3 b)
{
    return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}
