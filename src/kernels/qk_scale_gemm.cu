#include "src/kernels/qk_scale_gemm.h"
#include "src/core/check_cuda.h"
#include "src/kernels/_device_.cuh"

namespace cudaTransformer {

template <typename T>
__global__ void qkScaleGemmKernel(Tensor3D_Kernel<T> q, Tensor3D_Kernel<T> k, Tensor3D_Kernel<T> qk, const T scale) {
    int batch = blockDim.z * blockIdx.z + threadIdx.z;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y; 

    const int block_size = 8;

    Tensor3D_Kernel<T> qk_block = blockTensor(qk, 
                                            blockIdx.z, 
                                            blockIdx.x, 
                                            blockIdx.y, 
                                            block_size);

    T acc_sum = (T)0;

    for (int i = 0; i < k.row_size / block_size; i++) {
        Tensor3D_Kernel<T> q_block = blockTensor(q, blockIdx.z, blockIdx.x, i, block_size);
        Tensor3D_Kernel<T> k_block = blockTensor(k, blockIdx.z, i, blockIdx.y, block_size);

        __shared__ T q_shmem[block_size][block_size][block_size];
        __shared__ T k_shmem[block_size][block_size][block_size];

        if (batch < qk.batch_size && row < qk.col_size && col < qk.row_size) {
            q_shmem[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor(q_block, threadIdx.z, threadIdx.x, threadIdx.y);
            k_shmem[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor(k_block, threadIdx.z, threadIdx.x, threadIdx.y);

            __syncthreads();

            for (int j = 0; j < block_size; j++) {
                acc_sum += q_shmem[threadIdx.z][threadIdx.x][j] * k_shmem[threadIdx.z][j][threadIdx.y];
            }

            __syncthreads();
        }
    }

    if (batch < qk.batch_size && row < qk.col_size && col < qk.row_size)
        setValTensor(qk_block, threadIdx.z, threadIdx.x, threadIdx.y, acc_sum / scale);
}

template <typename T>
void qkScaleGemm(const Tensor<T> *q, const Tensor<T> *k, Tensor<T> *qk, const T scale) {
    Tensor3D_Kernel<T> _q(*q);
    Tensor3D_Kernel<T> _k(*k);
    Tensor3D_Kernel<T> _qk(*qk);


    int dim_x = _qk.col_size;
    int dim_y = _qk.row_size;
    int dim_z = _qk.batch_size;
    
    dim3 block(8, 8, 1);
    dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1, dim_z / block.z);
    qkScaleGemmKernel<T><<<grid, block>>>(_q, _k, _qk, scale);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void qkScaleGemm<float>(const Tensor<float> *q, const Tensor<float> *k, Tensor<float> *qk, const float scale);

}