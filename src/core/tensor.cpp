/**
 * Tensor.cc
 * 基本张量数据结构，承载运算数据的基本数据结构
 */

#include "src/core/tensor.h"

#include <stdio.h>

#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cnpy.h"
#include "src/core/check_cuda.h"
#include "src/core/mem_ops.h"
#include "src/kernels/kernel_utils.h"

namespace cudaTransformer {

template <typename T>
Tensor<T>::Tensor() {
    datatype = FP32;
    memtype = CPU;
    shape = {};
    data = nullptr;
    mem_pool = nullptr;
    static_mem = false;
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> _shape, MemoryPool<T>* _mem_pool, Datatype _datatype,
                  Memtype _memtype)
    : datatype(_datatype), mem_pool(_mem_pool), memtype(_memtype), shape(_shape) {
        if (mem_pool) {
            
           if (mem_pool->memtype != memtype) ERRORLOG("Memory Type Error: type of tensor is not same with memory pool.");

           data = mem_pool->dynamic_allocate(this->size());

        } else {
            if (memtype == CPU) {

                hostMalloc(data, this->size());

            } else if (memtype == GPU) {

                deviceMalloc(data, this->size());

            } else if (memtype == CPU_Pinned) {

                hostPinnedMem(data, this->size());

            } else if (memtype == UNI_MEM) {

                deviceManageMem(data, this->size());
            }
        }
}

template <typename T>
Tensor<T>::Tensor(Tensor &&tensor) {
    shape = tensor.shape;
    datatype = tensor.datatype;
    memtype = tensor.memtype;
    mem_pool = tensor.mem_pool;
    static_mem = tensor.static_mem;
    data = tensor.data;
    tensor.data = nullptr;
}

template <typename T>
Tensor<T>::Tensor(const Tensor &tensor, bool clone):
datatype(tensor.datatype), memtype(tensor.memtype), shape(tensor.shape), 
mem_pool(tensor.mem_pool), static_mem(tensor.static_mem)
{
    if (mem_pool) {

        data = mem_pool->dynamic_allocate(this->size());

        if (tensor.memtype == CPU) {

            H2Hcpy(this->data, tensor.data, tensor.size());

        } else if (tensor.memtype == GPU) {

            cudaD2Dcpy(this->data, tensor.data, tensor.size());

        } else if (tensor.memtype == CPU_Pinned) {

            H2Hcpy(this->data, tensor.data, tensor.size());

        } else if (tensor.memtype == UNI_MEM) {

            H2Hcpy(this->data, tensor.data, tensor.size());
        }

    } else {
        if (tensor.memtype == CPU) {

            hostMalloc(this->data, tensor.size());

            H2Hcpy(this->data, tensor.data, tensor.size());

        } else if (tensor.memtype == GPU) {

            deviceMalloc(this->data, tensor.size());

            cudaD2Dcpy(this->data, tensor.data, tensor.size());

        } else if (tensor.memtype == CPU_Pinned) {

            hostPinnedMem(this->data, tensor.size());

            H2Hcpy(this->data, tensor.data, tensor.size());

        } else if (tensor.memtype == UNI_MEM) {

            deviceManageMem(this->data, tensor.size());

            H2Hcpy(this->data, tensor.data, tensor.size());
        }
    }
}

template <typename T>
Tensor<T>::Tensor(const Tensor &tensor, Copytype cptype) {
    if (tensor.memtype == CPU_Pinned || tensor.memtype == UNI_MEM) {

        ERRORLOG(
            "Memor Type Error: location can not be changed in type of "
            "CPU_Pinned and UNI_MEM.");

    }

    this->datatype = tensor.datatype;

    this->shape = tensor.shape;

    if (cptype == GPU2CPU) {
        if (tensor.memtype == CPU) {
            ERRORLOG("Memory Type Error: same location.");
        }

        this->memtype = CPU;

        hostMalloc(this->data, tensor.size());

        cudaD2Hcpy(this->data, tensor.data, tensor.size());

    } else {
        if (tensor.memtype == GPU) {
            ERRORLOG("Memory Type Error: same location.");
        }

        deviceMalloc(this->data, tensor.size());

        cudaH2Dcpy(this->data, tensor.data, tensor.size());
    }
}

template <typename T>
Tensor<T>::Tensor(const cnpy::NpyArray &numpy_data, MemoryPool<T>* _mem_pool, Datatype _datatype,
                  Memtype _memtype, bool _static_mem):
                  shape(numpy_data.shape), memtype(_memtype), 
                  datatype(_datatype), mem_pool(_mem_pool), static_mem(_static_mem) 
{

    if (mem_pool) {

        if (mem_pool->memtype != memtype) ERRORLOG("Memory Type Error: type of tensor is not same with memory pool.");

        if (static_mem) 

            data = mem_pool->static_allocate(this->size());

        else 

            data = mem_pool->dynamic_allocate(this->size());

        if (memtype == CPU) {

            H2Hcpy(data, numpy_data.data<T>(), numpy_data.num_vals);

        } else if (memtype == GPU) {

            cudaH2Dcpy(data, numpy_data.data<T>(), numpy_data.num_vals);

        } else if (memtype == CPU_Pinned) {

            H2Hcpy(data, numpy_data.data<T>(), numpy_data.num_vals);

        } else if (memtype == UNI_MEM) {
            
            H2Hcpy(data, numpy_data.data<T>(), numpy_data.num_vals);
        }

    } else {
        if (memtype == CPU) {

            hostMalloc(data, numpy_data.num_vals);

            H2Hcpy(data, numpy_data.data<T>(), numpy_data.num_vals);

        } else if (memtype == GPU) {

            deviceMalloc(data, numpy_data.num_vals);

            cudaH2Dcpy(data, numpy_data.data<T>(), numpy_data.num_vals);

        } else if (memtype == CPU_Pinned) {

            hostPinnedMem(data, numpy_data.num_vals);

            H2Hcpy(data, numpy_data.data<T>(), numpy_data.num_vals);

        } else if (memtype == UNI_MEM) {
            
            deviceManageMem(data, numpy_data.num_vals);

            H2Hcpy(data, numpy_data.data<T>(), numpy_data.num_vals);
        }
    }
    
}

template <typename T>
Tensor<T>::Tensor(T *cpu_data, std::vector<size_t> _shape, MemoryPool<T>* _mem_pool, Datatype _datatype,
                  Memtype _memtype): shape(_shape), mem_pool(_mem_pool), memtype(_memtype), datatype(_datatype) 
{
    if (mem_pool) {

        if (mem_pool->memtype != memtype) ERRORLOG("Memory Type Error: type of tensor is not same with memory pool.");

        data = mem_pool->dynamic_allocate(this->size());

        if (memtype == GPU) {

            cudaH2Dcpy(data, cpu_data, this->size());

        } else if (memtype == CPU_Pinned) {

            H2Hcpy(data, cpu_data, this->size());

        } else if (memtype == UNI_MEM) {

            H2Hcpy(data, cpu_data, this->size());
        }

    } else {
        if (memtype == GPU) {

            deviceMalloc(data, this->size());

            cudaH2Dcpy(data, cpu_data, this->size());

        } else if (memtype == CPU_Pinned) {

            hostPinnedMem(data, this->size());

            H2Hcpy(data, cpu_data, this->size());

        } else if (memtype == UNI_MEM) {

            deviceManageMem(data, this->size());

            H2Hcpy(data, cpu_data, this->size());
        }
    }
    
}

template <typename T>
Tensor<T>::~Tensor() {
    if (data == nullptr) return; 
    if (mem_pool) {

        if (static_mem)
            mem_pool->static_deallocate(data, this->size());
        else
            mem_pool->dynamic_deallocate(data, this->size());

    } else {
        if (memtype == CPU) {

            hostFree(data);

        } else if (memtype == GPU) {

            deviceFree(data);

        } else if (memtype == CPU_Pinned) {

            hostPinFree(data);

        } else if (memtype == UNI_MEM) {

            deviceFree(data);

        }
    }
}
    

template <typename T>
size_t Tensor<T>::size() const {

    if (shape.size() == 0) {
        return 0;
    }

    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

template <typename T>
size_t Tensor<T>::sizeByte() const {return this->size() * sizeof(T);}

template <typename T>
void Tensor<T>::to(Memtype where) {
    if (where == CPU_Pinned || where == UNI_MEM) {

        ERRORLOG(
            "Memory Type Error: CPU_Pinned and UNI_MEM can not be carried.");

    } else if (where == CPU && memtype == GPU) {

        T *cpu_data;

        cudaD2Hcpy(cpu_data, data, this->size());

        deviceFree(data);

        if (data == nullptr) {
            data = cpu_data;
            memtype = CPU;
        }
    } else if (where == GPU && memtype == CPU) {

        T *gpu_data;

        cudaH2Dcpy(gpu_data, data, this->size());

        hostFree(data);

        if (data == nullptr) {
            data = gpu_data;
            memtype = GPU;
        }
    } else {
        ERRORLOG("Memory Type Error: Data is located at the same place.");
    }
}

template <typename T>
void printf_data(T *data, const std::vector<size_t> &shape) {
    printf("data : \n");
    // if (size > 5) size = 5;
    if (shape.size() == 2) {
        for (size_t row = 0; row < shape[0]; row++) {
            std::cout << "[ ";

            for (size_t col = 0; col < shape[1]; col++) {
                std::cout << data[row * shape[1] + col] << ", ";
            }
            std::cout << " ]" << std::endl;
        }
    } else if (shape.size() == 3) {
        for (size_t batch = 0; batch < shape[0]; batch++) {
            std::cout << "--------- batch:" << batch << " ---------"
                      << std::endl;
            for (size_t row = 0; row < shape[1]; row++) {
                std::cout << "[ ";
                for (size_t col = 0; col < shape[2]; col++) {
                    std::cout << data[batch * shape[1] * shape[2] +
                                      row * shape[2] + col]
                              << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }
}
template void printf_data(float *data, const std::vector<size_t> &shape);

template <typename T>
void Tensor<T>::show(bool show_val) const {

    CHECK(cudaDeviceSynchronize());

    std::string s_datatype, s_memtype;
    if (datatype == FP32)

        s_datatype = "FP32";

    else if (datatype == FP16)

        s_datatype = "FP16";

    else if (datatype == INT8)

        s_datatype = "INT8";

    if (memtype == CPU)

        s_memtype = "CPU";

    else if (memtype == GPU)

        s_memtype = "GPU";

    else if (memtype == CPU_Pinned)

        s_memtype = "CPU Pinned ";

    else if (memtype == UNI_MEM)

        s_memtype = "Unified";

    std::string s_shape = "[";

    for (size_t i = 0; i < shape.size() - 1; i++) {
        s_shape.append(std::to_string(shape[i]));
        s_shape.append(", ");
    }

    s_shape.append(std::to_string(shape[shape.size() - 1]));
    s_shape.push_back(']');

    printf("Tensor: { shape: %s, data type: %s, memory type: %s } ",
           s_shape.c_str(), s_datatype.c_str(), s_memtype.c_str());

    if (show_val) {
        if (memtype == GPU) {

            T *cpu_data;

            hostMalloc(cpu_data, this->size());

            cudaD2Hcpy(cpu_data, data, this->size());

            printf_data(cpu_data, shape);

            hostFree(cpu_data);

        } else {
            printf_data(data, shape);
        }

    } else
        std::cout << std::endl;
}

template <typename T>
void Tensor<T>::set_val(T val) {
    if (memtype == GPU) {

        setVal(this, val);
        
    } else {

        hostFill(data, this->size(), val);

    }
}

template <typename T>
void Tensor<T>::save_npy(const char *path) const {
    if (memtype == GPU) {

        T *cpu_data;

        hostMalloc(cpu_data, this->size());

        cudaD2Hcpy(cpu_data, data, this->size());

        cnpy::npy_save(path, cpu_data, shape, "w");

        hostFree(cpu_data);
    } else {
        cnpy::npy_save(path, data, shape, "w");
    }
}

template class Tensor<float>;

template <typename T>
Tensor3D_Kernel<T>::Tensor3D_Kernel(const Tensor<T> &tensor) {
    row_size = tensor.shape[2];
    col_size = tensor.shape[1];
    batch_size = tensor.shape[0];
    data = tensor.data;
}

template <typename T>
Tensor2D_Kernel<T>::Tensor2D_Kernel(const Tensor<T> &tensor) {
    row_size = tensor.shape[1];
    col_size = tensor.shape[0];
    data = tensor.data;
}

template struct Tensor3D_Kernel<float>;
template struct Tensor2D_Kernel<float>;

}  // namespace cudaTransformer