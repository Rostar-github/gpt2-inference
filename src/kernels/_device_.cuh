#pragma once
#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
__device__ T getValTensor(const Tensor3D_Kernel<T> &tensor, int batch, int row, int col) {
    return tensor.data[tensor.row_size * tensor.col_size * batch + tensor.row_size * row + col];
}

template <typename T>
__device__ void setValTensor(Tensor3D_Kernel<T> &tensor, int batch, int row, int col, T val) {
    tensor.data[tensor.row_size * tensor.col_size * batch + tensor.row_size * row + col] = val;
}

template <typename T>
__device__ Tensor3D_Kernel<T> blockTensor(const Tensor3D_Kernel<T> &tensor, int batch_block, int row_block, int col_block, const int block_size) {
    Tensor3D_Kernel<T> block_tensor;
    block_tensor.row_size = tensor.row_size;
    block_tensor.col_size = tensor.col_size;
    block_tensor.batch_size = tensor.batch_size;
    block_tensor.data = &tensor.data[block_size * (tensor.row_size * tensor.col_size * batch_block + tensor.row_size * row_block + col_block)];
    return block_tensor;
}

// template <typename T>
// __device__ void blockGemm(const Tensor3D_Kernel<T> tensor3d_x, 
//                     const Tensor3D_Kernel<T> tensor3d_y, 
//                     Tensor3D_Kernel<T> tensor3d_z,
//                     const int block_size) {

//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     int batch = blockIdx.z * blockDim.z + threadIdx.z;

//     Tensor3D_Kernel<T> block_tensor3d_z = blockTensor3D(tensor3d_z, blockIdx.z, blockIdx.x, blockIdx.y);


//     T acc_sum = (T)0;

//     for (int i = 0; i < tensor3d_y.col_size / block_size; i++) {
//         Tensor3D_Kernel<T> block_tensor3d_x = blockTensor3D(tensor3d_x, blockIdx.z, blockIdx.x, i);
//         Tensor3D_Kernel<T> block_tensor3d_y = blockTensor3D(tensor3d_y, blockIdx.z, i, blockIdx.y);
        
//         // block multiply
//         __shared__ T shared_block3d_x[1][block_size][block_size];
//         __shared__ T shared_block3d_y[1][block_size][block_size];

//         if (batch < tensor3d_z.batch_size && row < tensor3d_z.col_size && col < tensor3d_z.row_size) {
//             shared_block3d_x[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor3D(block_tensor3d_x, threadIdx.z, threadIdx.x, threadIdx.y);
//             shared_block3d_y[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor3D(block_tensor3d_y, threadIdx.z, threadIdx.x, threadIdx.y);

//             __syncthreads();

//             for (int j = 0; j < block_size; j++) {
//                 acc_sum += shared_block3d_x[threadIdx.z][threadIdx.x][j] * shared_block3d_y[threadIdx.z][j][threadIdx.y];
//             }
//             __syncthreads();
//         }
//     }
// }

}