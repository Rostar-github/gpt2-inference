#include "src/layers/self_attention.h"
#include "src/kernels/gemm.h"
#include "src/kernels/transpose.h"
#include "src/kernels/qkv_add_bias.h"
#include "src/kernels/qk_scale_gemm.h"
#include "src/kernels/softmax.h"

namespace cudaTransformer {

template <typename T>
void SelfAttention<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    q = new Tensor<T>({batch_size, n_samples, per_head_size}, mem_pool);
    k = new Tensor<T>({batch_size, n_samples, per_head_size}, mem_pool);
    v = new Tensor<T>({batch_size, n_samples, per_head_size}, mem_pool);

    transpose_k = new Tensor<T>({batch_size, per_head_size, n_samples}, mem_pool);

    qk = new Tensor<T>({batch_size, n_samples, n_samples}, mem_pool);

    softmax_qk = new Tensor<T>({batch_size, n_samples, n_samples}, mem_pool);


}

template <typename T>
void SelfAttention<T>::freeBuffer() {
    delete softmax_qk;
    delete qk;
    delete transpose_k;
    delete v;
    delete k;
    delete q;
}

template <typename T>
void SelfAttention<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    // input embedded hidden layer after layernorm = (batch, n_samples, hidden_size)
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[1];       // number of samples

    allocateBuffer(batch_size, n_samples);

    std::string w_q_name = "transformer.h.0.attn.q_" + std::to_string(head_index) + ".weight";
    std::string w_k_name = "transformer.h.0.attn.k_" + std::to_string(head_index) + ".weight";
    std::string w_v_name = "transformer.h.0.attn.v_" + std::to_string(head_index) + ".weight";

    std::string bias_q_name = "transformer.h.0.attn.q_" + std::to_string(head_index) + ".bias";
    std::string bias_w_k_name = "transformer.h.0.attn.k_" + std::to_string(head_index) + ".bias";
    std::string bias_w_v_name = "transformer.h.0.attn.v_" + std::to_string(head_index) + ".bias";


    Tensor<float>* w_q = weights->get(w_q_name);
    Tensor<float>* w_k = weights->get(w_k_name);
    Tensor<float>* w_v = weights->get(w_v_name);

    Tensor<float>* bias_q = weights->get(bias_q_name);
    Tensor<float>* bias_k = weights->get(bias_w_k_name);
    Tensor<float>* bias_v = weights->get(bias_w_v_name);

    gemm(input, w_q, q);
    gemm(input, w_k, k);
    gemm(input, w_v, v);

    qkvAddBias(q, k, v, bias_q, bias_k, bias_v);

    
    transpose(k, transpose_k);
    qkScaleGemm(q, transpose_k, qk, scale);
    softmax(qk, softmax_qk, n_samples);
    gemm(softmax_qk, v, output);

    freeBuffer();

}

template class SelfAttention<float>;

}