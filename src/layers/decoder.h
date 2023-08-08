#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"
#include "src/layers/mh_attention.h"
#include "src/layers/feed_forward.h"

namespace cudaTransformer {

template <typename T>
class Decoder: public Module<T> {

private:
    
    size_t index;
    size_t hidden_size;
    size_t per_head_size;
    size_t head_num;
    size_t ffinner_size;
    T epsilon;

    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;

    MultiHeadAttention<T>* multihead_atten;
    FeedForward<T>* feed_forward;

    Tensor<T>* atten_input;
    Tensor<T>* atten_output;
    Tensor<T>* feed_input;

    
protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;
    void construct_layers();

public:
    Decoder(
        size_t index,
        size_t hidden_size,
        size_t per_head_size,
        size_t head_num,
        size_t ffinner_size,
        T epsilon,
        TensorMap<T>* weights,
        MemoryPool<T>* mem_pool,
        bool free_buf_after_forward=true):
    Module<T>(free_buf_after_forward),
    index(index),
    hidden_size(hidden_size),
    per_head_size(per_head_size),
    head_num(head_num),
    ffinner_size(ffinner_size),
    epsilon(epsilon),
    weights(weights), 
    mem_pool(mem_pool){
        construct_layers();
    }

    virtual ~Decoder();

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}