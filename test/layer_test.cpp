#include "src/layers/embedding.h"
#include "src/layers/decoder.h"
#include "src/layers/lm_head.h"
#include <cuda_runtime.h>
#include <cmath>
#include <time.h>

using namespace cudaTransformer;

int main(int argc, char const *argv[]) {   
    cudaSetDevice(1);
    /*------------------------------------------------------------------------------*/
    // size_t batch_size = 1;
    // size_t max_seq_len = 1024;
    // size_t vocab_size = 50257;
    // size_t hidden_size = 768;

    // size_t n_samples = 10;

    // MemoryPool<float>* mem_pool = new MemoryPool<float>(50257 * 800 * 10, GPU);

    // TensorMap<float>* weights = new TensorMap<float>(mem_pool);

    // weights->load("/wanglina/cuda/cuda_example/model/gpt2.npz", FP32, GPU);

    // Embedding<float>* embedding_layer = new Embedding<float>(max_seq_len, vocab_size, hidden_size, weights, mem_pool);
        
    // float tokens[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // Tensor<float>* input = new Tensor<float>(tokens, {batch_size, 1, n_samples}, mem_pool);
    // Tensor<float>* output = new Tensor<float>({batch_size, n_samples, hidden_size}, mem_pool);

    // embedding_layer->forward(input, output);

    // output->save_npy();

    // delete output, input, embedding_layer, weights, mem_pool;

    /*------------------------------------------------------------------------------*/
    size_t batch_size = 1;
    size_t max_seq_len = 1024;
    size_t vocab_size = 50257;
    size_t hidden_size = 768;
    size_t per_head_size = 64;
    size_t head_num = 12;
    size_t ffinner_size = 3072;

    size_t n_samples = 10;

    MemoryPool<float>* mem_pool = new MemoryPool<float>(50257 * 800 * 10, GPU);

    TensorMap<float>* weights = new TensorMap<float>(mem_pool);

    weights->load("/wanglina/cuda/cuda_example/model/gpt2.npz", FP32, GPU);

    Embedding<float>* embedding_layer = new Embedding<float>(max_seq_len, vocab_size, hidden_size, weights, mem_pool);
    Decoder<float>* decoder = new Decoder<float>(0, hidden_size, per_head_size, head_num, ffinner_size, 1e-5, weights, mem_pool);
    LMHead<float>* lm_head = new LMHead<float>(vocab_size, hidden_size, 1e-5, weights, mem_pool);

        
    float tokens[10] = {0, 1, 2, 43, 14, 5, 14, 7, 8, 9};
    Tensor<float>* input = new Tensor<float>(tokens, {batch_size, 1, n_samples}, mem_pool);
    Tensor<float>* hidden = new Tensor<float>({batch_size, n_samples, hidden_size}, mem_pool);
    Tensor<float>* decoder_output = new Tensor<float>({batch_size, n_samples, hidden_size}, mem_pool);
    Tensor<float>* output = new Tensor<float>({batch_size, 1}, mem_pool);

    clock_t start, end;

    start = clock();

    embedding_layer->forward(input, hidden);

    // hidden->save_npy();

    decoder->forward(hidden, decoder_output);

    decoder->forward(decoder_output, hidden);

    decoder->forward(hidden, decoder_output);

    decoder->forward(decoder_output, hidden);

    decoder->forward(hidden, decoder_output);

    decoder->forward(decoder_output, hidden);

    decoder->forward(hidden, decoder_output);

    // decoder_output->save_npy();

    lm_head->forward(decoder_output, output);

    cudaDeviceSynchronize();

    end = clock();

    std::cout << (double)(end - start)/CLOCKS_PER_SEC << std::endl;

    output->save_npy();

    delete output, decoder_output, hidden, input, lm_head, decoder, embedding_layer, weights, mem_pool;

    return 0;
}
