#include <stdio.h>
#include <iostream>
#include "src/core/tensor.h"
#include "src/kernels/gemm.h"
#include "src/kernels/transpose.h"
#include "src/kernels/input_encoder.h"
#include <cuda_runtime.h>
#include "src/core/mem_pool.h"

using namespace cudaTransformer;

int main(int argc, char const *argv[])
{   
    cudaSetDevice(1);
    /*-------------------------Embedding Kernels Test Case--------------------*/
    // // MemoryPool<float>* mem_pool = new MemoryPool<float>(50257 * 800, GPU);
    // float tokens[5] = {0, 1, 2, 3, 4};
    // Tensor<float>* input = new Tensor<float>(tokens, {1, 1, 5});
    // Tensor<float>* onehot = new Tensor<float>({1, 5, 50257});
    // Tensor<float>* posi_code = new Tensor<float>({1, 5, 1024});
    // Tensor<float>* token_embed_w = new Tensor<float>({1, 50257, 768});
    // Tensor<float>* posi_embed_w = new Tensor<float>({1, 1024, 768});
    // Tensor<float>* output = new Tensor<float>({1, 5, 768});

    // token_embed_w->set_val(1.0f);
    // posi_embed_w->set_val(1.0f);

    // oneHotEncoder(input, onehot);
    // positionEncoder(posi_code);
    // embeddingEncoder(onehot, posi_code, token_embed_w, posi_embed_w, output);

    // // input->save_npy();
    // // posi_embed_w->show(true);
    // output->save_npy();
    // // posi_code->save_npy();
    // delete output;
    // delete posi_embed_w;
    // delete token_embed_w;
    // delete posi_code;
    // delete onehot;
    // delete input;
    // // delete mem_pool;
    /*-----------------------------------------------------------------------------*/
    MemoryPool<float>* mem_pool = new MemoryPool<float>(1024 * 1000, GPU);
    Tensor<float>* x = new Tensor<float>({1, 65, 128}, mem_pool);
    Tensor<float>* w_q = new Tensor<float>({1, 128, 64}, mem_pool);
    // Tensor<float>* w_k = new Tensor<float>({1, 128, 64});
    // Tensor<float>* w_v = new Tensor<float>({1, 128, 64});
    Tensor<float>* output = new Tensor<float>({1, 65, 64}, mem_pool);

    x->set_val(1.0f);
    w_q->set_val(1.0f);
    // w_k->set_val(1.0f);
    // w_v->set_val(1.0f);

    gemm(x, w_q, output);

    output->save_npy();

    delete output, w_q, x, mem_pool;

    return 0;
}
