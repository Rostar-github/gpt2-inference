#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"

namespace cudaTransformer {

template <typename T>
class LayerNorm: public Module<T> {

private:
    
    T epsilon;
    size_t decoder_index;
    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;

protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;

public:
    LayerNorm(
        T epsilon,
        size_t head_index,
        TensorMap<T>* weights, 
        MemoryPool<T>* mem_pool,
         bool free_buf_after_forward=true):
    Module<T>(free_buf_after_forward), 
    epsilon(epsilon),
    decoder_index(head_index),
    weights(weights), 
    mem_pool(mem_pool){

    }

    virtual ~LayerNorm() = default;

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}