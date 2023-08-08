#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"

namespace cudaTransformer {

template <typename T>
class LMHead: public Module<T> {

private:

    size_t vocab_size;
    size_t hidden_size;
    T epsilon;

    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;

    Tensor<T>* ln_output;
    Tensor<T>* transpose_proj;
    Tensor<T>* proj_output;
    Tensor<T>* softmax_output;
    Tensor<T>* max_val;
    

protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;

public:
    LMHead(
        size_t vocab_size,
        size_t hidden_size,
        T epsilon,
        TensorMap<T>* weights, 
        MemoryPool<T>* mem_pool, 
        bool free_buf_after_forward=true):
    Module<T>(free_buf_after_forward), 
    vocab_size(vocab_size),
    hidden_size(hidden_size),
    epsilon(epsilon),
    weights(weights), 
    mem_pool(mem_pool){

    }

    virtual ~LMHead() = default;

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}