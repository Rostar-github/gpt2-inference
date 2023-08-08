#include "src/models/gpt/gpt.h"
#include <cuda_runtime.h>

namespace cudaTransformer {

template <typename T>
void GPT2<T>::construct_layers() {
    // init module
    embedding = new Embedding<T>(max_seq_len, vocab_size, hidden_size, model_weights, mem_pool);
    
    for (size_t i = 0; i < decoder_num; i++) {
        decoders.push_back(new Decoder<T>(
            i,
            hidden_size,
            per_head_size,
            head_num,
            ffinner_size,
            epsilon,
            model_weights,
            mem_pool
        ));
    }

    lm_head = new LMHead<T>(vocab_size, hidden_size, epsilon, model_weights, mem_pool);
}

template <typename T>
void GPT2<T>::load(const char* model_path) {
    model_weights->load(model_path, FP32, GPU);
    construct_layers();
}


template <typename T>
void GPT2<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    // allocate intermediate result
    embed_out_buff = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);
    decoders_out_buff = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);
    
}

template <typename T>
void GPT2<T>::freeBuffer() {
    // free intermediate result
    delete decoders_out_buff;
    delete embed_out_buff;
}


template <typename T>
void GPT2<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[2];       // number of samples

    allocateBuffer(batch_size, n_samples);
    
    embedding->forward(input, embed_out_buff);

    cudaDeviceSynchronize();

    Tensor<T>* decoder_in = embed_out_buff;
    Tensor<T>* decoder_out = decoders_out_buff;

    for (size_t i = 0; i < decoder_num; i++) {
        decoders[i]->forward(decoder_in, decoder_out);
        cudaDeviceSynchronize();
        if (i != decoder_num - 1) {
            Tensor<T>* tmp = decoder_in;
            decoder_in = decoder_out;
            decoder_out = tmp;
        }
    }

    decoder_out->save_npy();

    lm_head->forward(decoder_out, output);


    freeBuffer();

}

template <typename T>
void GPT2<T>::operator()(T* tokens, size_t batch_size, size_t n_samples) {
    
    Tensor<T>* input = new Tensor<T>(tokens, {batch_size, 1, n_samples}, mem_pool);
    Tensor<T>* output = new Tensor<T>({batch_size, 1}, mem_pool);

    this->forward(input, output);

    // output->save_npy();

    delete output, input;
}


template <typename T>
GPT2<T>::~GPT2(){
    delete embedding;
    for (size_t i = 0; i < decoder_num; i++) {
        delete decoders.back();
        decoders.pop_back();
    }
    delete lm_head;

    // delete model_weights;
    delete mem_pool;
}

template class GPT2<float>;

}