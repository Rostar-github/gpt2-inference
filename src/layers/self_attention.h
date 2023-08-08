#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"

namespace cudaTransformer {

template <typename T>
class SelfAttention: public Module<T> {

private:

    size_t head_index;
    size_t hidden_size;
    size_t per_head_size;
    T scale;
    
    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;

    Tensor<float>* q;
    Tensor<float>* k;
    Tensor<float>* v;
    Tensor<float>* transpose_k;
    Tensor<float>* qk;
    Tensor<float>* softmax_qk;

protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;

public:
    SelfAttention(
        size_t head_index,
        size_t hidden_size,
        size_t per_head_size,
        T scale,
        TensorMap<T>* weights, 
        MemoryPool<T>* mem_pool, 
        bool free_buf_after_forward=true):
    Module<T>(free_buf_after_forward), 
    head_index(head_index),
    hidden_size(hidden_size), 
    per_head_size(per_head_size), 
    scale(scale),
    weights(weights), 
    mem_pool(mem_pool){}

    virtual ~SelfAttention() = default;

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}