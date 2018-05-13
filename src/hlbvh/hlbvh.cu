#include "hlbvh.h"
#include <cub/cub/cub.cuh>
#include "cuda_runtime_api.h"
#include "cuda.h"
#define BLOCK_SIZE 512

__device__ static
uint getGlobalIdx3DZXY()
{
    uint blockId = blockIdx.x
             + blockIdx.y * gridDim.x
             + gridDim.x * gridDim.y * blockIdx.z;
    return blockId * (blockDim.x * blockDim.y * blockDim.z)
              + (threadIdx.z * (blockDim.x * blockDim.y))
              + (threadIdx.y * blockDim.x)
              + threadIdx.x;
}

dim3 gridConfigure(uint64_t problemSize, dim3 block) {
    /// TODO
    /*dim3 MaxGridDim = {(uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[0],
                       (uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[1],
                       (uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[2]};
    dim3 gridDim = {1, 1, 1};

    uint64_t blockSize = block.x * block.y * block.z;

    if (problemSize > MaxGridDim.y * MaxGridDim.x * blockSize) {
        gridDim.z = problemSize / MaxGridDim.x * MaxGridDim.y * blockSize;
        problemSize = problemSize % MaxGridDim.x * MaxGridDim.y * blockSize;
    }

    if (problemSize > MaxGridDim.x * blockSize) {
        gridDim.y = problemSize / MaxGridDim.x * blockSize;
        problemSize = problemSize % MaxGridDim.x * blockSize;
    }

    gridDim.x = (problemSize + blockSize - 1) / blockSize;*/

    return dim3((problemSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

HLBVH::HLBVH() {
    this->memorySize = 0;
    this->numBVHLevels = 0;
    this->numNodes = 0;
    this->numReferences = 0;
    this->aabbMax = nullptr;
    this->aabbMin = nullptr;
    this->links = nullptr;
    this->memory = nullptr;
    this->parents = nullptr;
    this->references = nullptr;
    this->ranges = nullptr;
    builded = false;
}

bool HLBVH::alloc(uint32_t size) {
    memorySize = 2 * size * (2 * sizeof(int) + sizeof(uint2) + 2 * sizeof(float3)) + 1 * size * ( sizeof(uint) );
    cudaMalloc((void **)&memory, memorySize);
    if (memory == nullptr) {
        memorySize = 0;
        return false;
    }
    links =      reinterpret_cast<int*>    (memory);
    parents =    reinterpret_cast<int*>    (memory + 2 * size * sizeof(int));
    ranges =     reinterpret_cast<uint2*>  (memory + 2 * size * (2 * sizeof(int)));
    aabbMin =    reinterpret_cast<float3*> (memory + 2 * size * (2 * sizeof(int) + sizeof(uint2)));
    aabbMax =    reinterpret_cast<float3*> (memory + 2 * size * (2 * sizeof(int) + sizeof(uint2) + 1 * sizeof(float3)));
    references = reinterpret_cast<uint*>   (memory + 2 * size * (2 * sizeof(int) + sizeof(uint2) + 2 * sizeof(float3)));

    numBVHLevels = 0;
    numNodes = 0;
    numReferences = size;
    return true;
}

void HLBVH::free() {
    if (memory) {
        cudaFree((void *)memory);
    }
    builded = false;
    memory = nullptr;
    links = nullptr;
    parents = nullptr;
    ranges = nullptr;
    aabbMin = nullptr;
    aabbMax = nullptr;
    references = nullptr;
    memorySize = 0;
    numBVHLevels = 0;
    numNodes = 0;
    numReferences = 0;
}

__global__ static
void computeMortonCodesAndReferenceKernel(uint32_t *keys, uint *values, AABB *aabb, AABB globalAABB, uint32_t size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    keys[thread] = aabb[thread].getMortonCode(globalAABB);
    values[thread] = thread;
}

__global__ static
void swapKeysAndValues(uint32_t *keys, uint *valus, uint32_t *sortedKeys, uint *sortedValues, uint size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }
    keys[thread] = sortedKeys[thread];
    valus[thread] = sortedValues[thread];
}

void computeMortonCodesAndReference(uint32_t *keys, uint *values, AABB *aabb, AABB globalAABB, uint size) {
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    computeMortonCodesAndReferenceKernel <<<grid,block>>> (keys, values, aabb, globalAABB, size);
}

bool sortMortonCodes(uint32_t *keys, uint *values, uint size) {
    size_t allocSize = 0;
    uint32_t *keysOut = GpuStackAllocator::getInstance().alloc<uint32_t>(size);
    uint32_t *valuesOut = GpuStackAllocator::getInstance().alloc<uint32_t>(size);
    cub::DeviceRadixSort::SortPairs(nullptr, allocSize, keys, keysOut, values, valuesOut, size);

    uint8_t *tmpBuffer = GpuStackAllocator::getInstance().alloc<uint8_t>(allocSize);
    if (tmpBuffer == nullptr) {
        return false;
    }
    cub::DeviceRadixSort::SortPairs(tmpBuffer, allocSize, keys, keysOut, values, valuesOut, size);

    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    swapKeysAndValues<<<grid, block>>>(keys, values, keysOut, valuesOut, size);

    GpuStackAllocator::getInstance().free(tmpBuffer);
    GpuStackAllocator::getInstance().free(valuesOut);
    GpuStackAllocator::getInstance().free(keysOut);
    return true;
}

template<char comp, bool min>
__global__
void copyAABBComponent(float *dst, AABB *aabb, uint32_t size) {
    uint thread = getGlobalIdx3DZXY();
    if (thread >= size) {
        return;
    }

    switch(comp) {
        case 'x':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].x);
            } else {
                dst[thread] = AABBmax(aabb[thread].x);
            }
        }
        break;
        case 'y':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].y);
            } else {
                dst[thread] = AABBmax(aabb[thread].y);
            }
        }
        break;
        case 'z':
        {
            if (min) {
                dst[thread] = AABBmin(aabb[thread].z);
            } else {
                dst[thread] = AABBmax(aabb[thread].z);
            }
        }
        break;
    }
}

bool computeGlobalAABB(AABB *aabb, uint32_t size, AABB &result) {
    float *array = GpuStackAllocator::getInstance().alloc<float>(size);
    float *minmax = GpuStackAllocator::getInstance().alloc<float>(6);
    float cpuMinMax[6];
    size_t cub_tmp_memory_size = 0;


    cub::DeviceReduce::Min(nullptr, cub_tmp_memory_size, array, minmax, size);
    uint8_t *cub_tmp_memory = GpuStackAllocator::getInstance().alloc<uint8_t>(cub_tmp_memory_size);

    if (array == nullptr || minmax == nullptr || cub_tmp_memory == nullptr) {
        return false;
    }

    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid = gridConfigure(size, block);
    copyAABBComponent<'x', true> <<<grid, block>>> (array, aabb, size);
    cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 0, size);
    copyAABBComponent<'y', true> <<<grid, block>>> (array, aabb, size);
    cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 1, size);
    copyAABBComponent<'z', true> <<<grid, block>>> (array, aabb, size);
    cub::DeviceReduce::Min(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 2, size);
    copyAABBComponent<'x', false> <<<grid, block>>> (array, aabb, size);
    cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 3, size);
    copyAABBComponent<'y', false> <<<grid, block>>> (array, aabb, size);
    cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 4, size);
    copyAABBComponent<'z', false> <<<grid, block>>> (array, aabb, size);
    cub::DeviceReduce::Max(cub_tmp_memory, cub_tmp_memory_size, array, minmax + 5, size);

    cudaMemcpy(cpuMinMax, minmax, sizeof(float) * 6, cudaMemcpyDeviceToHost);

    result.x.x = cpuMinMax[0];
    result.y.x = cpuMinMax[1];
    result.z.x = cpuMinMax[2];

    result.x.y = cpuMinMax[3];
    result.y.y = cpuMinMax[4];
    result.z.y = cpuMinMax[5];
    GpuStackAllocator::getInstance().free(cub_tmp_memory);
    GpuStackAllocator::getInstance().free(minmax);
    GpuStackAllocator::getInstance().free(array);
    return true;
}

struct WorkQueue {
    int *nodeId;
    uint2 *range;
};

void initQueue(WorkQueue &queue, uint32_t size) {
    int nodeId = 0;
    uint2 range;
    range.x = 0;
    range.y = size;
    cudaMemcpy(queue.nodeId, &nodeId, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(queue.range, &range, sizeof(uint2), cudaMemcpyHostToDevice);
}



#define clz(x) __clz((x))
#define clzHost(x) ((x) == 0)? 32 : __builtin_clzll((x))
#define maxComp(x , y) ((x) > (y))? (x) : (y)
__global__
void split(HLBVH hlbvh, uint32_t *keys, uint queueSize, uint nodeSize, WorkQueue qIn, WorkQueue qOut, uint *counter) {
    uint thread = getGlobalIdx3DZXY();
    bool isWorking = true;
    if (thread >= queueSize) {
        isWorking = false;
    }

    int warpId = threadIdx.x / 32;
    __shared__  cub::WarpReduce<int>::TempStorage warpReduceTemp[BLOCK_SIZE / 32];
    __shared__  cub::WarpScan<int>::TempStorage warpScanTemp[BLOCK_SIZE / 32];
    uint2 rangeLeft, rangeRight, range;
    bool isLeaf = true;
    int node = 0;
    if (isWorking) {
        range = qIn.range[thread];
        rangeRight = range;
        rangeLeft = range;
        node = qIn.nodeId[thread];
        if (getRangeSize(range) > nodeSize) {
            isLeaf = false;
            uint32_t keyA = keys[getLeftBound(range)];
            uint32_t keyB = keys[getRightBound(range) - 1];
            uint32_t ha = 32;
            ha = clz(keyA ^ keyB);

            if (ha == 32) {
                uint mid = getLeftBound(range) + (getRightBound(range) - getLeftBound(range)) / 2;
                getRightBound(rangeLeft) = getLeftBound(rangeRight) = mid;
            } else {
                uint64_t mask = 1ULL << (32 - ha - 1);
                uint left, right;
                left = getLeftBound(range);
                right = getRightBound(range);
                bool test;
                uint mid;

                while (left < right) {
                    mid = left + (right - left) / 2;
                    test = (keys[mid] & mask) > 0;
                    /* key[mid] > key[left] */
                    if (test) {
                        right = mid;
                    } else {
                        left = mid + 1;
                    }
                }
                getRightBound(rangeLeft) = left;
                getLeftBound(rangeRight) = left;
            }
        }
    }

    int thread_data = 2;
    if (isLeaf) {
        thread_data = 0;
    }

    int sum = cub::WarpReduce<int>(warpReduceTemp[warpId]).Sum(thread_data);
    uint offset = 0;

    if ((threadIdx.x & 31) == 0 && isWorking) {
        offset = atomicAdd(counter, sum);
        thread_data += offset;
    }

    cub::WarpScan<int>(warpScanTemp[warpId]).ExclusiveSum(thread_data, thread_data);
    offset += thread_data;

    if (isWorking) {
        hlbvh.ranges[node] = range;
        if (isLeaf) {
            hlbvh.links[node] = LEAF;
        } else {
            uint left = hlbvh.numNodes + offset;
            hlbvh.links[node] = left;

            qOut.nodeId[offset] = left;
            qOut.nodeId[offset + 1] = left + 1;
            qOut.range[offset] = rangeLeft;
            qOut.range[offset + 1] = rangeRight;
        }
    }
}

bool buildTreeStructure(HLBVH &hlbvh, uint nodeSize, uint32_t *keys, uint *offset, uint32_t size) {
    WorkQueue work[2];
    work[0].nodeId = GpuStackAllocator::getInstance().alloc<int>(size);
    work[0].range = GpuStackAllocator::getInstance().alloc<uint2>(size);
    work[1].nodeId = GpuStackAllocator::getInstance().alloc<int>(size);
    work[1].range = GpuStackAllocator::getInstance().alloc<uint2>(size);
    offset[hlbvh.numBVHLevels++] = 0;
    uint* counter = GpuStackAllocator::getInstance().alloc<uint>(400);
    if (work[0].nodeId == nullptr || work[0].range == nullptr
        || work[1].nodeId == nullptr || work[1].range == nullptr
        || counter == nullptr)
    {
        return false;
    }

    int switcher = 0;
    cudaMemset(counter, 0, sizeof(uint) * 400);
    initQueue(work[switcher], size);
    uint queueSize = 1;
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid;

    while(queueSize > 0) {
        grid = gridConfigure(queueSize, block);
        split<<<grid, block>>>(hlbvh, keys, queueSize, nodeSize, work[switcher], work[1 - switcher], counter);
        cudaMemcpy(&queueSize, counter, sizeof(uint), cudaMemcpyDeviceToHost);
        counter++;
        hlbvh.numNodes += queueSize;
        switcher = 1 - switcher;
        if (queueSize > 0) {
            offset[hlbvh.numBVHLevels++] = hlbvh.numNodes;
        }
    }
    GpuStackAllocator::getInstance().free(counter);
    GpuStackAllocator::getInstance().free(work[1].range);
    GpuStackAllocator::getInstance().free(work[1].nodeId);
    GpuStackAllocator::getInstance().free(work[0].range);
    GpuStackAllocator::getInstance().free(work[0].nodeId);
    return true;
}
/// } Build Tree topology
/////////////////////////////////////////////
/// Refit boxes {
__host__ __device__
void readBoxFromAABB(AABB *aabb, uint id, float3 &min, float3 &max) {
    AABB read = aabb[id];
    min.x = read.x.x;
    min.y = read.y.x;
    min.z = read.z.x;

    max.x = read.x.y;
    max.y = read.y.y;
    max.z = read.z.y;
}

__global__
void refitBoxesKernel(HLBVH hlbvh, AABB *aabb, uint2 range, bool isRoot) {
    uint node = getGlobalIdx3DZXY();
    if (node >= getRangeSize(range)) {
        return;
    }

    node += getLeftBound(range);
    int link = hlbvh.links[node];
    if (link == LEAF) {
        uint2 nodeRange = hlbvh.ranges[node];
        float3 aabbMin;
        float3 aabbMax;
        readBoxFromAABB(aabb, hlbvh.references[getLeftBound(nodeRange)], aabbMin, aabbMax);
        for (uint j = getLeftBound(nodeRange) + 1; j < getRightBound(nodeRange); j++) {
            float3 readAABBMin;
            float3 readAABBMax;
            readBoxFromAABB(aabb, hlbvh.references[j], readAABBMin, readAABBMax);
            aabbMin = fmin(aabbMin, readAABBMin);
            aabbMax = fmax(aabbMax, readAABBMax);
        }
        hlbvh.aabbMin[node] = aabbMin;
        hlbvh.aabbMax[node] = aabbMax;
    } else {
        float3 boxAaabbMin = hlbvh.aabbMin[link + 0];
        float3 boxAaabbMax = hlbvh.aabbMax[link + 0];

        float3 boxBaabbMin = hlbvh.aabbMin[link + 1];
        float3 boxBaabbMax = hlbvh.aabbMax[link + 1];

        hlbvh.parents[link + 0] = node;
        hlbvh.parents[link + 1] = node;

        hlbvh.aabbMin[node] = fmin(boxAaabbMin, boxBaabbMin);
        hlbvh.aabbMax[node] = fmax(boxAaabbMax, boxBaabbMax);
    }

    if (isRoot) {
        hlbvh.parents[node] = -1;
    }
}

void refitBoxes(HLBVH &hlbvh, AABB *aabb, uint *offset) {
    bool isRoot;
    dim3 block = dim3(BLOCK_SIZE);
    dim3 grid;
    for (int i = hlbvh.numBVHLevels - 2; i >= 0; --i)  {
        uint2 range;
        getLeftBound(range) = offset[i];
        getRightBound(range) = offset[i + 1];
        if (i == 0) {
            isRoot = true;
        }
        grid = gridConfigure(getRangeSize(range), block);
        refitBoxesKernel<<<grid, block>>>(hlbvh, aabb, range, isRoot);
    }
}
/// } Refit boxes
/////////////////////////////////////////////

bool HLBVH::build(AABB *aabb, uint32_t size) {
    if (size == 0) { return false; }

    if (alloc(size) == false) {
        return false;
    }

    GpuStackAllocator::getInstance().pushPosition();
    do {
        AABB globalAABB;
        if (!computeGlobalAABB(aabb, size, globalAABB)) {
            break;
        }
        auto keys = GpuStackAllocator::getInstance().alloc<uint32_t>(size);
        uint offset[400];
        if (keys == nullptr || offset == nullptr) {
            break;
        }
        computeMortonCodesAndReference(keys, references, aabb, globalAABB, size);
        if (!sortMortonCodes(keys, references, size)) {
            break;
        }
        if (!buildTreeStructure(*this, 1, keys, offset, size)) {
            break;
        }
        refitBoxes(*this, aabb, offset);
        builded = true;
        GpuStackAllocator::getInstance().popPosition();
        return true;
    } while (0);
    GpuStackAllocator::getInstance().popPosition();
    return false;
}
