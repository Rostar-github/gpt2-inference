#include "src/models/gpt/gpt.h"
#include <cuda_runtime.h>
#include <time.h>

using namespace cudaTransformer;

int main(int argc, char const *argv[])
{
    cudaSetDevice(1);
    size_t batch_size = 1;
    size_t max_seq_len = 1024;
    size_t vocab_size = 50257;
    size_t hidden_size = 768;
    size_t per_head_size = 64;
    size_t head_num = 12;
    size_t decoder_num = 12;
    size_t ffinner_size = 3072;

    size_t n_samples = 10;

    GPT2<float>* model = new GPT2<float>(
        max_seq_len,
        head_num,
        per_head_size,
        ffinner_size,
        decoder_num,
        vocab_size,
        hidden_size,
        1e-5f
    );

    model->load("/wanglina/cuda/cuda_example/model/gpt2.npz");

    float tokens[32] = {8241, 318, 262, 1301, 30, 23, 14, 244, 355, 654, 7457, 4235, 43 , 55, 64, 12,
                        231, 124, 333, 444, 646, 77, 88, 123, 323, 111, 888, 9999, 7767, 3214, 2321, 1111};

    clock_t start, end;

    start = clock();

    (*model)(tokens, batch_size, n_samples);

    cudaDeviceSynchronize();

    end = clock();

    std::cout << (double)(end - start)/CLOCKS_PER_SEC << std::endl;

    delete model;
    return 0;
}
