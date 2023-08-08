#pragma once

#include "src/layers/module.h"
#include "src/core/tensor.h"
#include "src/core/tensor_map.h"
#include "src/core/mem_pool.h"

namespace cudaTransformer {

template <typename T>
class Embedding: public Module<T> {

private:
    /*------------------------------------config param-------------------------------- */
    size_t max_seq_len;
    size_t vocab_size;
    size_t hidden_size;
    /*-----------------------------------------------------------------------------------*/
    TensorMap<T>* weights;
    MemoryPool<T>* mem_pool;
    /*-----------------------------------------buffer-------------------------------------*/
    Tensor<T>* onehot;
    Tensor<T>* posi_code;

protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;

public:
    Embedding(
        size_t max_seq_len, 
        size_t vocab_size, 
        size_t hidden_size, 
        TensorMap<T>* weights, 
        MemoryPool<T>* mem_pool,
        bool free_buf_after_forward = true):
    Module<T>(free_buf_after_forward),
    max_seq_len(max_seq_len), 
    vocab_size(vocab_size), 
    hidden_size(hidden_size), 
    weights(weights),
    mem_pool(mem_pool){}

    virtual ~Embedding() = default;

    void forward(Tensor<T> *input, Tensor<T> *output);

}; 


}