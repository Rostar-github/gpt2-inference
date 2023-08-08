#pragma once

#include "src/layers/module.h"
#include "src/core/tensor_map.h"
#include "src/layers/embedding.h"
#include "src/layers/decoder.h"
#include "src/core/mem_pool.h"
#include "src/layers/lm_head.h"
// #include "src/core/allocator.h"
#include <stdio.h>
#include <vector>

namespace cudaTransformer {

template <typename T>
class GPT2: public Module<T> {

private:
    /*------------------------------------config param-------------------------------- */
    size_t max_seq_len;     // max length of sentence                       default: 1024
    size_t head_num;        // number of heads in multi-head attention      default: 12
    size_t per_head_size;   // size of output tensor through a head         default: 64
    size_t ffinner_size;    // inner size of feed forward layer             default: 3072
    size_t decoder_num;     // number for decoders                          default: 12
    size_t vocab_size;      // size of vocabulary                           default: 50257
    size_t hidden_size;     // size of hidden layer                         default: 768
    T epsilon;
    /*-----------------------------------------------------------------------------------*/
    TensorMap<T>* model_weights;
    MemoryPool<T>* mem_pool;
    /*-------------------------------------define layers---------------------------------*/
    Embedding<T>* embedding;
    std::vector<Decoder<T>*> decoders;
    LMHead<T>* lm_head;
    /*------------------------------------------------------------------------------------*/
    /*-----------------------------------------buffer-------------------------------------*/
    Tensor<T>* embed_out_buff;
    Tensor<T>* decoders_out_buff;


protected:
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) override;
    virtual void freeBuffer() override;
    void construct_layers();  // config, construct modules and load module params
    
public: 
    GPT2(
        size_t max_seq_len,
        size_t head_num,
        size_t per_head_size,
        size_t ffinner_size,
        size_t decoder_num,
        size_t vocab_size,
        size_t hidden_size,
        T epsilon,
        bool free_buf_after_forward=true
    ): Module<T>(free_buf_after_forward), 
    max_seq_len(max_seq_len), 
    head_num(head_num),
    per_head_size(per_head_size),
    ffinner_size(ffinner_size), 
    decoder_num(decoder_num), 
    vocab_size(vocab_size), 
    hidden_size(hidden_size),
    epsilon(epsilon){
        mem_pool = new MemoryPool<T>(50257 * 100 * 45, GPU);  
        model_weights = new TensorMap<T>(mem_pool);
    };

    virtual ~GPT2();

    void load(const char* model_path);

    void forward(Tensor<T> *input, Tensor<T> *output);

    void operator()(T* tokens, size_t batch_size, size_t n_samples);
};


}