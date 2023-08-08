/**
 * TensorMap.cc
 * 存储模型参数的基本数据结构，由unordered_map<string, Tensor>组成
 */
#include "src/core/tensor_map.h"

#include <vector>

#include "cnpy.h"
#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
TensorMap<T>& TensorMap<T>::load(const char* path, Datatype load_dtype, Memtype load_dev) {
    try {
        /* code */
        cnpy::npz_t params_map = cnpy::npz_load(path);
        for (auto &params : params_map) {
            std::string name = params.first;
            cnpy::NpyArray npy_weights = params.second;
            layer_weights.insert(
                std::pair<std::string, Tensor<T>>(
                    name, 
                    Tensor<T>(npy_weights, mem_pool, load_dtype, load_dev, true)));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::cerr << "Model load failed..." << '\n';
    }

    return *this;
}

template <typename T>
Tensor<T>* TensorMap<T>::get(std::string name) {
    return &layer_weights[name];
}

template class TensorMap<float>;
}  // namespace cudaTransformer