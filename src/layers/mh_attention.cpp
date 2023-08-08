#include "src/layers/mh_attention.h"
#include "src/kernels/concate_heads.h"
#include "src/kernels/gemm.h"
#include <cmath>

namespace cudaTransformer {

template <typename T>
void MultiHeadAttention<T>::construct_layers() {
    for (size_t i = 0; i < head_num; i++) {
        attention_heads.push_back(
            new SelfAttention<T>(i, hidden_size, per_head_size, 
            sqrtf(hidden_size), weights, mem_pool)
        );
    }

    
}

template <typename T>
void MultiHeadAttention<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    for (size_t i = 0; i < head_num; i++) {
        atten_outputs.push_back(new Tensor<T>({batch_size, n_samples, per_head_size}, mem_pool));
    }
    concated_atten = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);

}

template <typename T>
void MultiHeadAttention<T>::freeBuffer() {
    delete concated_atten;
    for (size_t i = 0; i < head_num; i++) {
        delete atten_outputs.back();
        atten_outputs.pop_back();
    }
}

template <typename T>
void MultiHeadAttention<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[1];       // number of samples

    allocateBuffer(batch_size, n_samples);

    /*--------------multi-head attention-------------*/

    std::string proj_name = "transformer.h."+ std::to_string(index) + ".attn.project.weight";

    Tensor<T>* project = weights->get(proj_name);

    for (size_t i = 0; i < head_num; i++) {
        attention_heads[i]->forward(input, atten_outputs[i]);
    }

    // AllGather OP
    concateHeads(concated_atten, atten_outputs);

    gemm(concated_atten, project, output);

    /*-----------------------------------------------*/

    output->save_npy();

    freeBuffer();

}

template <typename T>
MultiHeadAttention<T>::~MultiHeadAttention() {
    for (size_t i = 0; i < head_num; i++) {
        delete attention_heads.back();
        attention_heads.pop_back();
    }
}

template class MultiHeadAttention<float>;

}