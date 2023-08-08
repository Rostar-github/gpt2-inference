#include "src/layers/decoder.h"
#include "src/layers/embedding.h"
#include "src/layers/feed_forward.h"
#include "src/kernels/layer_norm.h"
#include "src/kernels/add_bias_residual_ln.h"
#include "src/kernels/add_bias_residual.h"

namespace cudaTransformer {

template <typename T>
void Decoder<T>::construct_layers() {
    multihead_atten = new MultiHeadAttention<T>(index, hidden_size,
                                                per_head_size, head_num, weights, mem_pool);
    feed_forward = new FeedForward<T>(index, ffinner_size, hidden_size, weights, mem_pool);
    
}

template <typename T>
void Decoder<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    atten_input = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);
    atten_output = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);
    feed_input = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);
}

template <typename T>
void Decoder<T>::freeBuffer() {
    delete feed_input;
    delete atten_output;
    delete atten_input;
}

template <typename T>
void Decoder<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[1];       // number of samples

    allocateBuffer(batch_size, n_samples);

    std::string alpha_name_1 = "transformer.h." + std::to_string(index) + ".ln_1.weight";
    std::string beta_name_1 = "transformer.h." + std::to_string(index) + ".ln_1.bias";

    Tensor<T>* alpha_1 = weights->get(alpha_name_1);
    Tensor<T>* beta_1 = weights->get(beta_name_1);

    layerNorm(input, atten_input, alpha_1, beta_1, epsilon);

    // atten_input->save_npy();

    multihead_atten->forward(atten_input, atten_output);

    // AddBiasResidualLayerNorm (add project bias)
    std::string bias_name = "transformer.h." + std::to_string(index) + ".attn.project.bias";
    std::string alpha_name_2 = "transformer.h." + std::to_string(index) + ".ln_2.weight";
    std::string beta_name_2 = "transformer.h." + std::to_string(index) + ".ln_2.bias";

    Tensor<T>* atten_proj_bias = weights->get(bias_name);
    Tensor<T>* alpha_2 = weights->get(alpha_name_2);
    Tensor<T>* beta_2 = weights->get(beta_name_2);

    // addBiasResidualLN(atten_output, feed_input, atten_proj_bias, 
    //                     input, alpha_2, beta_2, epsilon);

    addBiasResidual(atten_output, atten_proj_bias, input);

    layerNorm(atten_output, feed_input, alpha_2, beta_2, epsilon);

    // feed_forward->forward
    feed_forward->forward(feed_input, output);
        
    // AddBiasResidual (add project bias)
    std::string feed_bias_name = "transformer.h." + std::to_string(index) + ".mlp.project.bias";
    Tensor<T>* feed_bias = weights->get(feed_bias_name);

    addBiasResidual(output, feed_bias, atten_output);

    // output->save_npy();

    freeBuffer();

}

template <typename T>
Decoder<T>::~Decoder() {
    delete feed_forward;
    delete multihead_atten;
}

template class Decoder<float>;

}