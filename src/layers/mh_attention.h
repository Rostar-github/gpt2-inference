#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"
#include "src/layers/self_attention.h"
#include <vector>

namespace cudaTransformer {

template <typename T>
class MultiHeadAttention: public Module<T> {

private:

    size_t index;
    size_t hidden_size;
    size_t per_head_size;
    size_t head_num;

    
    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;

    std::vector<SelfAttention<T>*> attention_heads;

    std::vector<Tensor<T>*> atten_outputs;
    Tensor<T>* concated_atten;


protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;
    void construct_layers();

public:
    MultiHeadAttention(
        size_t index,
        size_t hidden_size,
        size_t per_head_size,
        size_t head_num,
        TensorMap<T>* weights, 
        MemoryPool<T>* mem_pool, 
        bool free_buf_after_forward=true):
    Module<T>(free_buf_after_forward), 
    index(index),
    hidden_size(hidden_size),
    per_head_size(per_head_size),
    head_num(head_num),
    weights(weights), 
    mem_pool(mem_pool){
        construct_layers();
    }

    virtual ~MultiHeadAttention();

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}