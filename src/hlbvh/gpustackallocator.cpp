#include "gpustackallocator.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
GpuStackAllocator::GpuStackAllocator() {
    this->m_memory = 0;
    this->m_size = 0;
    this->m_top = 0;
}

GpuStackAllocator::~GpuStackAllocator(){
     deinit();
}

void GpuStackAllocator::resize(uint64_t size) {
    if (m_memory != 0) {
        cudaFree((void *) m_memory);
        m_memory = 0;
    }
    cudaMalloc((void **)&m_memory, size);
    m_top = m_memory;
    m_size = size;
    m_alloced.clear();
}

void GpuStackAllocator::deinit() {
    if (m_memory) {
        cudaFree((void *) m_memory);
        m_alloced.clear();
        m_memory = 0;
        m_size = 0;
        m_top = 0;
    }
}

void GpuStackAllocator::pushPosition() {
    m_alloced.push_back(GpuStackBoarder);
}

bool GpuStackAllocator::popPosition() {
    if (m_alloced.empty()) {
        return false;
    }

    while (!m_alloced.empty() && m_alloced.back() != GpuStackBoarder) {
        m_top -= m_alloced.back();
        m_alloced.pop_back();
    }

    if (!m_alloced.empty()) {
        m_alloced.pop_back();
    }
    return true;
}

void GpuStackAllocator::clear() {
    m_top = m_memory;
    m_alloced.clear();
}
