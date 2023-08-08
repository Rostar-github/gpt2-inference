#include "src/layers/feed_forward.h"
#include "src/kernels/gemm.h"
#include "src/kernels/add_bias_gelu.h"

namespace cudaTransformer {

template <typename T>
void FeedForward<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    fc_output = new Tensor<T>({batch_size, n_samples, ffinner_size}, mem_pool);
    
}

template <typename T>
void FeedForward<T>::freeBuffer() {
    delete fc_output;
}

template <typename T>
void FeedForward<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[1];       // number of samples

    allocateBuffer(batch_size, n_samples);

    std::string fc_weight_name = "transformer.h." + std::to_string(index) + ".mlp.fc.weight";
    Tensor<T>* fc_weight = weights->get(fc_weight_name);
    
    gemm(input, fc_weight, fc_output);

    std::string fc_bias_name = "transformer.h." + std::to_string(index) + ".mlp.fc.bias";
    Tensor<T>* fc_bias = weights->get(fc_bias_name);

    // Add bias Gelu
    addBiasGeluInPlace(fc_output, fc_bias);

    std::string proj_weight_name = "transformer.h." + std::to_string(index) + ".mlp.project.weight";
    Tensor<T>* project = weights->get(proj_weight_name);

    gemm(fc_output, project, output);

    freeBuffer();

}

template class FeedForward<float>;

}