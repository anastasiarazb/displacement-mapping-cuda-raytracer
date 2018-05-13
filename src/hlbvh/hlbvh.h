#pragma once
#include "gpustackallocator.h"
#include "../cuda_numerics/float3.h"
#define LEAF 0xFFFFFFFF
#define getLeftBound(p) p.x
#define getRightBound(p) p.y
#define getRangeSize(p) (p.y - p.x)

#define AABBmin(p) p.x
#define AABBmax(p) p.y

class AABB {
public:
    float2 x;
    float2 y;
    float2 z;

    __device__ __host__
    int64_t clamp(int64_t in, int64_t min, int64_t max) {
        int64_t res = (in > max)? max : in;
        res = (res < min)? min : res;
        return res;
    }

    __device__ __host__
    uint32_t getMortonCode(AABB const &global) {
        float3 centroid;
        uint32_t code = 0;
        centroid.z = AABBmin(z) + 0.5 * (AABBmax(z) - AABBmin(z));
        centroid.x = AABBmin(x) + 0.5 * (AABBmax(x) - AABBmin(x));
        centroid.y = AABBmin(y) + 0.5 * (AABBmax(y) - AABBmin(y));

        centroid.x -= AABBmin(global.x);
        centroid.y -= AABBmin(global.y);
        centroid.z -= AABBmin(global.z);

        centroid.x /= (AABBmax(global.x) - AABBmin(global.x));
        centroid.y /= (AABBmax(global.y) - AABBmin(global.y));
        centroid.z /= (AABBmax(global.z) - AABBmin(global.z));

        uint3 centroidInt;
        const double shift = (double)(UINT_MAX);
        centroidInt.x = (uint32_t)clamp(centroid.x * shift, 0, UINT_MAX);
        centroidInt.y = (uint32_t)clamp(centroid.y * shift, 0, UINT_MAX);
        centroidInt.z = (uint32_t)clamp(centroid.z * shift, 0, UINT_MAX);

        uint p = 30;
        for (int i = 0; i < 10; i++) {
            code |= ((centroidInt.x & 0x80000000) >> 31) << (p    );
            code |= ((centroidInt.y & 0x80000000) >> 31) << (p - 1);
            code |= ((centroidInt.z & 0x80000000) >> 31) << (p - 2);
            p -= 3;
        }
        return code;
    }
};

class HLBVH {
public:
    bool build(AABB *aabb, uint32_t size);
    bool search(AABB aabb);
    bool isBuilded() { return builded; }
    HLBVH();
public:
    bool alloc(uint32_t size);
    void free();
    uint32_t numNodes;
    uint32_t numReferences;
    uint32_t numBVHLevels;
    float3 *aabbMin;
    float3 *aabbMax;

    uint *references;
    int *parents;
    uint2 *ranges;
    int *links;
    uint8_t *memory;
    uint64_t memorySize;
    bool builded;
};

