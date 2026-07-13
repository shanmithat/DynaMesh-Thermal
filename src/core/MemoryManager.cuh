#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"[CUDA GPU ERROR]: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace dynamesh {

template <typename T>
class ManagedAllocator {
public:
    static T* allocate(size_t size, bool useUnified = true) {
        if (size == 0) return nullptr;
        T* ptr = nullptr;
        if (useUnified) {
            gpuErrchk(cudaMallocManaged(&ptr, size * sizeof(T)));
        } else {
            gpuErrchk(cudaMalloc(&ptr, size * sizeof(T)));
        }
        return ptr;
    }

    static void deallocate(T* ptr) {
        if (ptr) {
            gpuErrchk(cudaFree(ptr));
        }
    }
};

// Smart pointer like wrapper for continuous unified/device memory blocks
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t size = 0, bool useUnified = true) 
        : size_(size), useUnified_(useUnified) {
        data_ = ManagedAllocator<T>::allocate(size_, useUnified_);
    }

    ~DeviceBuffer() {
        ManagedAllocator<T>::deallocate(data_);
    }

    // Disable copy for simplicity, allow move
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_), useUnified_(other.useUnified_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            ManagedAllocator<T>::deallocate(data_);
            data_ = other.data_;
            size_ = other.size_;
            useUnified_ = other.useUnified_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

private:
    T* data_;
    size_t size_;
    bool useUnified_;
};

} // namespace dynamesh
