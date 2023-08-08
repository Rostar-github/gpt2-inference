#include "src/layers/lm_head.h"
#include "src/kernels/layer_norm.h"
#include "src/kernels/transpose.h"
#include "src/kernels/gemm.h"
#include "src/kernels/softmax.h"
#include "src/kernels/gready_search.h"


namespace cudaTransformer {

template <typename T>
void LMHead<T>::allocateBuffer(size_t batch_size, size_t n_samples) {
    ln_output = new Tensor<T>({batch_size, n_samples, hidden_size}, mem_pool);
    transpose_proj = new Tensor<T>({1, hidden_size, vocab_size}, mem_pool);
    proj_output = new Tensor<T>({batch_size, n_samples, vocab_size}, mem_pool);
    softmax_output = new Tensor<T>({batch_size, n_samples, vocab_size}, mem_pool);
    max_val = new Tensor<T>({batch_size, 1}, mem_pool);
}

template <typename T>
void LMHead<T>::freeBuffer() {
    delete max_val;
    delete softmax_output;
    delete proj_output;
    delete transpose_proj;
    delete ln_output;
}

template <typename T>
void LMHead<T>::forward(Tensor<T> *input, Tensor<T> *output) {
    size_t batch_size = input->shape[0];      // batch size
    size_t n_samples = input->shape[1];       // number of samples

    allocateBuffer(batch_size, n_samples);

    std::string alpha_name = "transformer.ln_f.weight";
    std::string beta_name = "transformer.ln_f.bias";

    Tensor<T>* alpha = weights->get(alpha_name);
    Tensor<T>* beta = weights->get(beta_name);

    // input->save_npy();

    layerNorm(input, ln_output, alpha, beta, epsilon);

    // ln_output->save_npy();

    std::string proj_name = "lm_head";
    
    Tensor<T>* project = weights->get(proj_name);

    transpose(project, transpose_proj);

    gemm(ln_output, transpose_proj, proj_output);

    // proj_output->save_npy();

    softmax(proj_output, softmax_output, (int)vocab_size);

    // softmax_output->save_npy();

    greadySearch(softmax_output, output, max_val, n_samples, vocab_size);

    // max_val->save_npy();

    // output->save_npy();

    freeBuffer();

}

template class LMHead<float>;

}