#pragma once

#include <cuda_runtime_api.h>
#include <inttypes.h>
#define FUNC_PREFIX __host__ __device__

template<typename T>
class GpuStack {
public:
    FUNC_PREFIX
    GpuStack(T *memory, uint32_t size);

    FUNC_PREFIX
    T top();

    FUNC_PREFIX
    T &topRef();

    FUNC_PREFIX
    void pop();

    FUNC_PREFIX
    bool push(T elem);

    FUNC_PREFIX
    bool empty();
public:
    T *m_gpuMemory;
    uint32_t m_top;
    uint32_t m_size;
};

template<typename T> FUNC_PREFIX
GpuStack<T>::GpuStack(T *memory, uint32_t size) {
    if (memory == nullptr) {
        m_gpuMemory = nullptr;
        m_size = 0;
        m_top = 0;
    } else {
        m_gpuMemory = memory;
        m_size = size;
        m_top = 0;
    }
}

template<typename T> FUNC_PREFIX
bool GpuStack<T>::push(T elem) {
    if (m_top == m_size || m_gpuMemory == nullptr) {
        return false;
    }
    m_gpuMemory[m_top] = elem;
    ++m_top;
    return true;
}

template<typename T> FUNC_PREFIX
T GpuStack<T>::top() {
    return m_gpuMemory[m_top - 1];
}

template<typename T> FUNC_PREFIX
T &GpuStack<T>::topRef() {
    return m_gpuMemory[m_top - 1];
}

template<typename T> FUNC_PREFIX
void GpuStack<T>::pop() {
    if (m_top > 0) {
        --m_top;
    }
}

template<typename T> FUNC_PREFIX
bool GpuStack<T>::empty() {
    return m_top == 0;
}
