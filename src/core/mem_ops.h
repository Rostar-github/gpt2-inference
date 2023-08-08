#include <cuda_runtime.h>

#ifndef MEM_OPS_H
#define MEM_OPS_H

namespace cudaTransformer {

template <typename T>
void hostMalloc(T* &data_ptr, const size_t size);

template <typename T>
void hostPinnedMem(T* &data_ptr, const size_t size);

template <typename T>
void hostFree(T* &data_ptr);

template <typename T>
void hostPinFree(T* &data_ptr);

template <typename T>
void hostFill(T* data_ptr, const size_t size, T value);

template <typename T>
void deviceMalloc(T* &data_ptr, const size_t size);

template <typename T>
void deviceManageMem(T* &data_ptr, const size_t size);

template <typename T>
void deviceZeroMem(T* data_ptr, const size_t size);

template <typename T>
void deviceFill(T* dev_data_ptr, const size_t size, T value);

template <typename T>
void deviceFree(T* &dev_data_ptr);  // pointer reference

template <typename T>
void cudaD2Hcpy(T* target, const T* src, const size_t size);

template <typename T>
void cudaH2Dcpy(T* target, const T* src, const size_t size);

template <typename T>
void cudaD2Dcpy(T* target, const T* src, const size_t size);

template <typename T>
void H2Hcpy(T* target, const T* src, const size_t size);

}  // namespace cudaTransformer

#endif