#pragma once
#include "inttypes.h"
#include <deque>

class GpuStackAllocator {
public:
    const uint64_t GpuStackBoarder = 0xFFFFFFFFFFFFFFFFULL;
    const uint64_t GpuAlignSize = 256;

    virtual ~GpuStackAllocator();

    void resize(uint64_t size);

    inline static GpuStackAllocator& getInstance() {
        static GpuStackAllocator *gGpuStackAllocator = new GpuStackAllocator();
        static bool init = false;
        if (init == false) {
            init = true;
        }
        return *gGpuStackAllocator;
    }

    template<typename T>
    T* alloc(uint64_t count = 1) {
        uint64_t objSize = sizeof(T) * count;
        uint64_t offset = GpuAlignSize - 1;
        uint64_t newtop = m_top + offset + objSize;
        if (count != 0 && newtop <= m_memory + m_size) {
                uintptr_t aligned = (m_top + offset) & ~(GpuAlignSize - 1); // выравняли указатель
                m_top = m_top + offset + objSize; // сдвинули указатель на top стека
                m_alloced.push_back(objSize + offset);
                return reinterpret_cast<T*>(aligned);
        }
        //printf("nullptr was return! requested memory %zu | free %zu", offset + objSize, availableMemory());
        //exit(-1);
        return nullptr;
    }

    void clear();

    template<typename T>
    bool free(T* handle) {
        if (handle == nullptr) {
            return false;
        }

        while(!m_alloced.empty() && m_alloced.back() == GpuStackBoarder) {
            m_alloced.pop_back();
        }

        if (!m_alloced.empty()) {
            uint64_t size = m_alloced.back();
            uint64_t offset = GpuAlignSize - 1;
            uintptr_t aligned = (m_top - size + offset) & ~(GpuAlignSize - 1);
            if (aligned == reinterpret_cast<uintptr_t>(handle)) {
                m_top -= size;
                m_alloced.pop_back();
                return true;
            }
        }
        return false;
    }

    uint64_t availableMemory() {
        return m_memory + m_size - m_top;
    }

    void pushPosition();
    bool popPosition();
    void deinit();
private:
    GpuStackAllocator();
    std::deque<uint64_t> m_alloced; // используем как стек
    uint64_t m_size;
    uintptr_t m_memory;
    uintptr_t m_top;
};
