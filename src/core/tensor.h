#include <vector>
#include "cnpy.h"
#include "src/core/mem_pool.h"
#include "src/core/type.h"
// #include <cuda_runtime.h>

#ifndef TENSOR_H
#define TENSOR_H

namespace cudaTransformer {

template <typename T>
class Tensor {
private:
    MemoryPool<T>* mem_pool = nullptr;
    bool static_mem = false;
public:
    Datatype datatype;
    Memtype memtype;
    std::vector<size_t> shape;
    T* data;
    

    Tensor();

    Tensor(std::vector<size_t> _shape, MemoryPool<T>* _mem_pool=nullptr, Datatype _datatype=FP32,  Memtype _memtype=GPU);

    // move
    Tensor(Tensor &&tensor);

    Tensor(const Tensor &tensor) = default;

    // deep copy constructor [copy to new tensor]
    Tensor(const Tensor &tensor, bool clone);
    
    // deep copy constructor and change data location [copy to new tensor(other location)]
    Tensor(const Tensor &tensor, Copytype cptype);

    // construct Tensor from numpy data [cnpy]
    Tensor(const cnpy::NpyArray &numpy_data, MemoryPool<T>* _mem_pool=nullptr, Datatype _datatype=FP32, Memtype _memtype=GPU, bool _static_mem=false);

    // construct Tesnor from cpu data [array]
    Tensor(T* cpu_data, std::vector<size_t> _shape, MemoryPool<T>* _mem_pool=nullptr, Datatype _datatype=FP32, Memtype _memtype=GPU);

    ~Tensor();

    // get size of data == shape[0] * ... * shape[n-1]
    size_t size() const;

    // get byte size of data
    size_t sizeByte() const;

    // carry data from CPU to GPU or from GPU to CPU [in place]
    void to(Memtype where);

    // display propertise for debug
    void show(bool show_val=false) const;

    // save tensor as a .npz file for preview
    void save_npy(const char* path = "/wanglina/cuda/cuda_example/tools/tesnor.npy") const;

    // set Tensor value
    void set_val(T val);

};


template <typename T>
struct Tensor3D_Kernel {
    size_t row_size;
    size_t col_size;
    size_t batch_size;
    T* data;
    Tensor3D_Kernel() = default;
    Tensor3D_Kernel(const Tensor<T> &tensor);
}; 

template <typename T>
struct Tensor2D_Kernel {
    size_t row_size;
    size_t col_size;
    T* data;
    Tensor2D_Kernel() = default;
    Tensor2D_Kernel(const Tensor<T> &tensor);
}; 

}  // namespace cudaTransformer

#endif