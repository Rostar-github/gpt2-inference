#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"

namespace cudaTransformer {

template <typename T>
class FeedForward: public Module<T> {

private:
    
    size_t index;
    size_t ffinner_size;
    size_t hidden_size;

    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;

    Tensor<T>* fc_output;

protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;

public:
    FeedForward(
        size_t index,
        size_t ffinner_size,
        size_t hidden_size,
        TensorMap<T>* weights, 
        MemoryPool<T>* mem_pool, 
        bool free_buf_after_forward=true):
    Module<T>(free_buf_after_forward), 
    index(index),
    ffinner_size(ffinner_size),
    hidden_size(hidden_size),
    weights(weights), 
    mem_pool(mem_pool){

    }

    virtual ~FeedForward() = default;

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}