
#include <string>
#include <unordered_map>
#include "src/core/tensor.h"
#include "src/core/mem_pool.h"

#ifndef TENSORMAP_H
#define TENSORMAP_H

namespace cudaTransformer {

template <typename T>
class TensorMap {
private:
    std::unordered_map<std::string, Tensor<T>> layer_weights;
    MemoryPool<T>* mem_pool;
public:

    TensorMap(MemoryPool<T>* _mem_pool): mem_pool(_mem_pool){};

    TensorMap& load(const char* path, Datatype load_dtype, Memtype load_dev);

    Tensor<T>* get(std::string name);

};

}

#endif