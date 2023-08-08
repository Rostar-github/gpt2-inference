/**
 * memory_ops.cc
 * 封装主机内存操作和设备内存操作，（主要为了加上错误处理等操作）
*/
#include <algorithm>
#include <cuda_runtime.h>
#include <string.h>
#include "src/core/check_cuda.h"
#include "src/core/mem_ops.h"

namespace cudaTransformer {

template <typename T>
void hostMalloc(T* &data_ptr, const size_t size) {
    data_ptr = (T*)malloc(sizeof(T) * size);
}
template void hostMalloc<float>(float* &data_ptr, const size_t size);


template <typename T>
void hostPinnedMem(T* &data_ptr, const size_t size) {
    CHECK(cudaMallocHost((T**)&data_ptr, sizeof(T) * size, 0));
}
template void hostPinnedMem<float>(float* &data_ptr, const size_t size);

template <typename T>
void hostFree(T* &data_ptr) {
    if (data_ptr != nullptr) {
        free(data_ptr);
        data_ptr = nullptr;
    }
}
template void hostFree<float>(float* &data_ptr);

template <typename T>
void hostPinFree(T* &data_ptr) {
    if (data_ptr != nullptr) {
        CHECK(cudaFreeHost(data_ptr));
        data_ptr = nullptr;
    }
}
template void hostPinFree<float>(float* &data_ptr);


template <typename T>
void hostFill(T* data_ptr, const size_t size, T value) {
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    memcpy(data_ptr, arr, sizeof(T) * size);
    delete [] arr;
}
template void hostFill<float>(float* data_ptr, const size_t size, float value);

template<typename T>
void deviceMalloc(T* &data_ptr, const size_t size) {
    CHECK(cudaMalloc((T**)&data_ptr, sizeof(T) * size));
}
template void deviceMalloc<float>(float* &data_ptr, const size_t size);


template <typename T>
void deviceManageMem(T* &data_ptr, const size_t size) {
    CHECK(cudaMallocManaged((T**)&data_ptr, sizeof(T) * size));
}
template void deviceManageMem<float>(float* &data_ptr, const size_t size);


template<typename T>
void deviceZeroMem(T* data_ptr, const size_t size) {
    CHECK(cudaMemset(static_cast<void*>(data_ptr), 0, sizeof(T) * size));
}
template void deviceZeroMem<float>(float* data_ptr, const size_t size);


template<typename T>
void deviceFill(T* dev_data_ptr, const size_t size, T value) {
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    CHECK(cudaMemcpy(dev_data_ptr, arr, sizeof(T) * size, cudaMemcpyHostToDevice));
    delete[] arr;
}
template void deviceFill<float>(float* dev_data_ptr, const size_t size, float value);


template <typename T>
void deviceFree(T* &dev_data_ptr) {
    if (dev_data_ptr != nullptr){
        CHECK(cudaFree(dev_data_ptr));
        dev_data_ptr = nullptr;
    }
}
template void deviceFree<float>(float* &dev_data_ptr);


template <typename T>
void cudaD2Hcpy(T* target, const T* src, const size_t size) {
    CHECK(cudaMemcpy(target, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}
template void cudaD2Hcpy<float>(float* target, const float* src, const size_t size);


template <typename T>
void cudaH2Dcpy(T* target, const T* src, const size_t size) {
    CHECK(cudaMemcpy(target, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}
template void cudaH2Dcpy<float>(float* target, const float* src, const size_t size);


template <typename T>
void cudaD2Dcpy(T* target, const T* src, const size_t size) {
    CHECK(cudaMemcpy(target, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}
template void cudaD2Dcpy<float>(float* target, const float* src, const size_t size);


template <typename T>
void H2Hcpy(T* target, const T* src, const size_t size) {
    memcpy(target, src, sizeof(T) * size);
}
template void H2Hcpy<float>(float* target, const float* src, const size_t size);

}