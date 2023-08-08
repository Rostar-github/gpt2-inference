#include "src/core/check_cuda.h"
#include "src/kernels/input_encoder.h"
#include "src/kernels/_device_.cuh"

namespace cudaTransformer {

template <typename T>
__global__ void oneHotEncoderKernel(const Tensor3D_Kernel<T> tokens,
                                    Tensor3D_Kernel<T> onehot) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockDim.z * blockIdx.z + threadIdx.z;

    if (col < onehot.row_size) {
        int ids = tokens.data[batch * tokens.row_size + row];
        if (col == ids) {
            onehot.data[batch * onehot.col_size * onehot.row_size +
                        row * onehot.row_size + col] = (T)1;
        }
    }
}

template <typename T>
void oneHotEncoder(const Tensor<T>* tokens, Tensor<T>* onehot) {
    Tensor3D_Kernel<T> _tokens(*tokens);
    Tensor3D_Kernel<T> _onehot(*onehot);

    int dim_x = _onehot.col_size;
    int dim_y = _onehot.row_size;
    int dim_z = _onehot.batch_size;

    dim3 block(1, 512, 1);
    dim3 grid(dim_x / block.x, (dim_y - 1) / block.y + 1,
              (dim_z - 1) / block.z + 1);
    oneHotEncoderKernel<T><<<grid, block>>>(_tokens, _onehot);
    CHECK(cudaGetLastError());
}
template void oneHotEncoder<float>(const Tensor<float>* tokens,
                                   Tensor<float>* onehot);

template <typename T>
__global__ void positionEncoderKernel(Tensor3D_Kernel<T> posi_code) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockDim.z * blockIdx.z + threadIdx.z;

    int block_row = blockIdx.x;

    if (col < posi_code.row_size) {
        if (col == block_row) {
            posi_code.data[batch * posi_code.col_size * posi_code.row_size +
                           row * posi_code.row_size + col] = (T)1;
        }
    }
}

template <typename T>
void positionEncoder(Tensor<T>* posi_code) {
    Tensor3D_Kernel<T> _posi_code(*posi_code);

    int dim_x = _posi_code.col_size;
    int dim_y = _posi_code.row_size;
    int dim_z = _posi_code.batch_size;

    dim3 block(1, 512, 1);
    dim3 grid(dim_x / block.x, (dim_y - 1) / block.y + 1,
              (dim_z - 1) / block.z + 1);
    positionEncoderKernel<T><<<grid, block>>>(_posi_code);
    CHECK(cudaGetLastError());
}
template void positionEncoder<float>(Tensor<float>* posi_code);

template <typename T>
__global__ void embeddingEncoderKernel(Tensor3D_Kernel<T> onehot,
                                       Tensor3D_Kernel<T> posi_code,
                                       Tensor3D_Kernel<T> token_embed_w,
                                       Tensor3D_Kernel<T> posi_embed_w,
                                       Tensor3D_Kernel<T> hidden) 
{
    const int block_size = 16;

    int batch = blockDim.z * blockIdx.z + threadIdx.z;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    Tensor3D_Kernel<T> hidden_block = blockTensor(hidden, 
                                                blockIdx.z, 
                                                blockIdx.x, 
                                                blockIdx.y, 
                                                block_size);

    T sum_1 = (T)0;

    for (int i = 0; i < onehot.row_size / block_size; i++) {
        Tensor3D_Kernel<T> onehot_block = blockTensor(onehot, blockIdx.z, blockIdx.x, i, block_size);
        Tensor3D_Kernel<T> tew_block = blockTensor(token_embed_w, blockIdx.z, i, blockIdx.y, block_size);

        __shared__ T onehot_shmem[1][block_size][block_size];
        __shared__ T tew_shmem[1][block_size][block_size];

        if (batch < hidden.batch_size && row < hidden.col_size && col < hidden.row_size) {
            onehot_shmem[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor(onehot_block, threadIdx.z, threadIdx.x, threadIdx.y);
            tew_shmem[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor(tew_block, threadIdx.z, threadIdx.x, threadIdx.y);

            __syncthreads();

            for (int j = 0; j < block_size; j++) {
                sum_1 += onehot_shmem[threadIdx.z][threadIdx.x][j] * tew_shmem[threadIdx.z][j][threadIdx.y];
            }

            __syncthreads();
        }
    }

    T sum_2 = (T)0;

    for (int i = 0; i < posi_code.row_size / block_size; i++) {
        Tensor3D_Kernel<T> posi_block = blockTensor(posi_code, blockIdx.z, blockIdx.x, i, block_size);
        Tensor3D_Kernel<T> pew_block = blockTensor(posi_embed_w, blockIdx.z, i, blockIdx.y, block_size);

        __shared__ T posi_shmem[1][block_size][block_size];
        __shared__ T pew_shmem[1][block_size][block_size];

        if (batch < hidden.batch_size && row < hidden.col_size && col < hidden.row_size) {
            posi_shmem[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor(posi_block, threadIdx.z, threadIdx.x, threadIdx.y);
            pew_shmem[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor(pew_block, threadIdx.z, threadIdx.x, threadIdx.y);

            __syncthreads();

            for (int j = 0; j < block_size; j++) {
                sum_2 += posi_shmem[threadIdx.z][threadIdx.x][j] * pew_shmem[threadIdx.z][j][threadIdx.y];
            }

            __syncthreads();
        }
    }

    if (batch < hidden.batch_size && row < hidden.col_size && col < hidden.row_size) {
        setValTensor(hidden_block, threadIdx.z, threadIdx.x, threadIdx.y, sum_1 + sum_2);
    }
                               
}

template <typename T>
void embeddingEncoder(const Tensor<T>* onehot, const Tensor<T>* posi_code,
                      const Tensor<T>* token_embed_w,
                      const Tensor<T>* posi_embed_w, Tensor<T>* hidden) {

    Tensor3D_Kernel<T> _onehot(*onehot);
    Tensor3D_Kernel<T> _posi_code(*posi_code);
    Tensor3D_Kernel<T> _hidden(*hidden);
    Tensor3D_Kernel<T> _token_embed_w(*token_embed_w);
    Tensor3D_Kernel<T> _posi_embed_w(*posi_embed_w);

    int dim_x = _hidden.col_size;
    int dim_y = _hidden.row_size;
    int dim_z = _hidden.batch_size;

    dim3 block(16, 16, 1);
    dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1,
              dim_z / block.z);
    embeddingEncoderKernel<T><<<grid, block>>>(_onehot, _posi_code, _token_embed_w, _posi_embed_w, _hidden);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void embeddingEncoder<float>(const Tensor<float>* onehot,
                                      const Tensor<float>* posi_code,
                                      const Tensor<float>* token_embed_w,
                                      const Tensor<float>* posi_embed_w,
                                      Tensor<float>* hidden);

}  // namespace cudaTransformer