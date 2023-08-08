#include "src/layers/embedding.h"
#include "src/kernels/input_encoder.h"

namespace cudaTransformer {

template <typename T>
void Embedding<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    onehot = new Tensor<T>({batch_size, n_samples, vocab_size}, mem_pool);
    posi_code = new Tensor<T>({batch_size, n_samples, max_seq_len}, mem_pool);
}

template <typename T>
void Embedding<T>::freeBuffer() {
    delete posi_code;
    delete onehot;
}

template <typename T>
void Embedding<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[2];       // number of samples

    allocateBuffer(batch_size, n_samples);

    Tensor<T>* token_embed_w = weights->get("transformer.token_embed.weight");
    Tensor<T>* posi_embed_w = weights->get("transformer.posi_embed.weight");

    oneHotEncoder(input, onehot);
    positionEncoder(posi_code);

    embeddingEncoder(onehot, posi_code, token_embed_w, posi_embed_w, output);


    freeBuffer();

}

template class Embedding<float>;

}