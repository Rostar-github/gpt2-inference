#include "src/layers/ln_norm.h"
#include "src/kernels/layer_norm.h"

namespace cudaTransformer {

template <typename T>
void LayerNorm<T>::allocateBuffer(size_t batch_size, size_t n_samples) {

}


template <typename T>
void LayerNorm<T>::freeBuffer() {

}

template <typename T>
void LayerNorm<T>::forward(Tensor<T> *input, Tensor<T> *output) {


    std::string alpha_name = "transformer.h." + std::to_string(decoder_index) +".ln_1.weight";
    std::string beta_name = "transformer.h." + std::to_string(decoder_index) +".ln_1.bias";

    Tensor<T>* alpha = weights->get(alpha_name);
    Tensor<T>* beta = weights->get(beta_name);

    layerNorm(input, output, alpha, beta, epsilon);


}

template class LayerNorm<float>;

}