
#include "src/kernels/gemm.h"
#include "src/core/check_cuda.h"
// #include "cuda_runtime.h"

#define block_size 8

namespace cudaTransformer {

// int block_size = 8;

template <typename T>
__device__ T getValTensor2D(const Tensor2D_Kernel<T> &tensor, int row, int col) {
    return tensor.data[tensor.row_size * row + col];
}

template <typename T>
__device__ T getValTensor3D(const Tensor3D_Kernel<T> &tensor, int batch, int row, int col) {
    // printf("addr: %u \n", tensor.data);
    return tensor.data[tensor.row_size * tensor.col_size * batch + tensor.row_size * row + col];
}

template <typename T>
__device__ void setValTensor2D(Tensor2D_Kernel<T> &tensor, int row, int col, T val) {
    tensor.data[tensor.row_size * row + col] = val;
}

template <typename T>
__device__ void setValTensor3D(Tensor3D_Kernel<T> &tensor, int batch, int row, int col, T val) {
    tensor.data[tensor.row_size * tensor.col_size * batch + tensor.row_size * row + col] = val;
}

template <typename T>
__device__ Tensor3D_Kernel<T> blockTensor3D(const Tensor3D_Kernel<T> &tensor, int batch_block, int row_block, int col_block) {
    Tensor3D_Kernel<T> block_tensor;
    block_tensor.row_size = tensor.row_size;
    block_tensor.col_size = tensor.col_size;
    block_tensor.batch_size = 1;
    block_tensor.data = &tensor.data[block_size * (tensor.row_size * tensor.col_size * batch_block + tensor.row_size * row_block + col_block)];
    return block_tensor;
}

template <typename T>
__device__ Tensor2D_Kernel<T> blockTensor2D(const Tensor2D_Kernel<T> &tensor, int row_block, int col_block) {
    Tensor2D_Kernel<T> block_tensor;
    block_tensor.row_size = block_size;
    block_tensor.col_size = block_size;
    block_tensor.data = &tensor.data[block_size * (tensor.row_size * row_block + col_block)];
    return block_tensor;
}


template <typename T>
__global__ void gemmKernel(const Tensor3D_Kernel<T> tensor3d_x, const Tensor3D_Kernel<T> tensor3d_y, Tensor3D_Kernel<T> tensor3d_z) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    // const int block_size = 16;

    Tensor3D_Kernel<T> block_tensor3d_z = blockTensor3D(tensor3d_z, blockIdx.z, blockIdx.x, blockIdx.y);


    T acc_sum = (T)0;

    for (int i = 0; i < tensor3d_y.col_size / block_size; i++) {
        Tensor3D_Kernel<T> block_tensor3d_x = blockTensor3D(tensor3d_x, blockIdx.z, blockIdx.x, i);
        Tensor3D_Kernel<T> block_tensor3d_y = blockTensor3D(tensor3d_y, blockIdx.z, i, blockIdx.y);
        
        // block multiply
        __shared__ T shared_block3d_x[1][block_size][block_size];
        __shared__ T shared_block3d_y[1][block_size][block_size];

        // if (batch < tensor3d_z.batch_size && row < tensor3d_z.col_size && col < tensor3d_z.row_size) {
        shared_block3d_x[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor3D(block_tensor3d_x, threadIdx.z, threadIdx.x, threadIdx.y);
        shared_block3d_y[threadIdx.z][threadIdx.x][threadIdx.y] = getValTensor3D(block_tensor3d_y, threadIdx.z, threadIdx.x, threadIdx.y);
    
        __syncthreads();


        for (int j = 0; j < block_size; j++) {
            acc_sum += shared_block3d_x[threadIdx.z][threadIdx.x][j] * shared_block3d_y[threadIdx.z][j][threadIdx.y];
        }
        
        __syncthreads();
        // } 
    }
    
    if (batch < tensor3d_z.batch_size && row < tensor3d_z.col_size && col < tensor3d_z.row_size) {
        
        setValTensor3D(block_tensor3d_z, threadIdx.z, threadIdx.x, threadIdx.y, acc_sum);
    }
        
}


// 3D Tensor x 3D Tensor (when multiply output tensor with the other output tensor) tensor3d = (row , col , batch)
template <typename T>
void gemm(const Tensor<T> *tensor3d_x, const Tensor<T> *tensor3d_y, Tensor<T> *tensor3d_z) {
    //assert
    assert(tensor3d_x->shape[2] == tensor3d_y->shape[1]);
    // change data format from tensor to tensor under kernel
    Tensor3D_Kernel<T> kernel_tensor3d_x(*tensor3d_x);
    Tensor3D_Kernel<T> kernel_tensor3d_y(*tensor3d_y);
    Tensor3D_Kernel<T> kernel_tensor3d_z(*tensor3d_z);

    int dim_x = kernel_tensor3d_z.col_size;
    int dim_y = kernel_tensor3d_z.row_size;
    int dim_z = kernel_tensor3d_z.batch_size;

    // const int block_size = 16;

    dim3 block(block_size, block_size, 1);
    dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1, dim_z / block.z);
    gemmKernel<T><<<grid, block>>>(kernel_tensor3d_x, kernel_tensor3d_y, kernel_tensor3d_z);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void gemm<float>(const Tensor<float> *tensor3d_x, const Tensor<float> *tensor3d_y, Tensor<float> *tensor3d_z);







// template <typename T>
// __global__ void TensorMul2DKernel(const Tensor3D_Kernel<T> tensor3d_x, const Tensor2D_Kernel<T> tensor2d_y, Tensor3D_Kernel<T> tensor3d_z) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     int batch = blockIdx.z * blockDim.z + threadIdx.z;
//     T acc_sum = (T)0;
//     for (int i = 0; i < tensor2d_y.col_size; i++) {
//         acc_sum += getValTensor3D(tensor3d_x, batch, row, i) * getValTensor2D(tensor2d_y, i, col);
//     setValTensor3D(tensor3d_z, batch, row, col, acc_sum);
//     }
// }

// template <typename T>
// __global__ void TensorMul3DKernel(const Tensor3D_Kernel<T> tensor3d_x, const Tensor3D_Kernel<T> tensor3d_y, Tensor3D_Kernel<T> tensor3d_z) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     int batch = blockIdx.z * blockDim.z + threadIdx.z;
//     T acc_sum = (T)0;
//     for (int i = 0; i < tensor3d_y.col_size; i++) {
//         acc_sum += getValTensor3D(tensor3d_x, batch, row, i) * getValTensor3D(tensor3d_y, batch, i, col);
    
//     setValTensor3D(tensor3d_z, batch, row, col, acc_sum);
//     }
// }




// template <typename T>
// __global__ void gemm2DKernel(const Tensor3D_Kernel<T> tensor3d_x, const Tensor2D_Kernel<T> tensor2d_y, Tensor3D_Kernel<T> tensor3d_z) {
//     int block_row = blockIdx.x;
//     int block_col = blockIdx.y;
//     int block_batch = blockIdx.z;

//     Tensor3D_Kernel<T> block_tensor3d_z = blockTensor3D(tensor3d_z, block_batch, block_row, block_col);

//     int row = threadIdx.x;
//     int col = threadIdx.y;
//     int batch = threadIdx.z;

//     T acc_sum = (T)0;

//     for (int i = 0; i < tensor2d_y.col_size / block_size; i++) {
//         Tensor3D_Kernel<T> block_tensor3d_x = blockTensor3D(tensor3d_x, block_batch, block_row, i);
//         Tensor2D_Kernel<T> block_tensor2d_y = blockTensor2D(tensor2d_y, i, block_col);
        
//         // block multiply
//         __shared__ T shared_block3d_x[1][block_size][block_size];
//         __shared__ T shared_block2d_y[block_size][block_size];

//         if (batch < tensor3d_z.batch_size && row < tensor3d_z.col_size && col < tensor3d_z.row_size) {
//             shared_block3d_x[batch][row][col] = getValTensor3D(block_tensor3d_x, batch, row, col);
//             shared_block2d_y[row][col] = getValTensor2D(block_tensor2d_y, row, col);

//             __syncthreads();

//             for (int j = 0; j < block_size; j++) {
//                 acc_sum += shared_block3d_x[batch][row][j] * shared_block2d_y[j][col];
//             }
//             __syncthreads();
//         }
        
//     }
//     if (batch < tensor3d_z.batch_size && row < tensor3d_z.col_size && col < tensor3d_z.row_size)
//         setValTensor3D(block_tensor3d_z, batch, row, col, acc_sum);
// }


// // 3D Tensor x 2D Matrix (when multiply input tensor with layer weight) tensor3d = (row , col , batch)
// template <typename T>
// void gemm2D(const Tensor<T> *tensor3d_x, const Tensor<T> *tensor2d_y, Tensor<T> *tensor3d_z) {
//     // assert
//     assert(tensor3d_x->shape[2] == tensor2d_y->shape[0]);
//     // change data format from tensor to tensor under kernel
//     Tensor3D_Kernel<T> kernel_tensor3d_x(*tensor3d_x);
//     Tensor2D_Kernel<T> kernel_tensor2d_y(*tensor2d_y);
//     Tensor3D_Kernel<T> kernel_tensor3d_z(*tensor3d_z);

//     int dim_x = kernel_tensor3d_z.col_size;
//     int dim_y = kernel_tensor3d_z.row_size;
//     int dim_z = kernel_tensor3d_z.batch_size;
//     // const int block_size = 8;

//     dim3 block(block_size, block_size, 1);
//     dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1, dim_z / block.z);
//     gemm2DKernel<T><<<grid, block>>>(kernel_tensor3d_x, kernel_tensor2d_y, kernel_tensor3d_z);
//     CHECK(cudaGetLastError());
    
// }
// template void gemm2D<float>(const Tensor<float> *tensor3d_x, const Tensor<float> *tensor2d_y, Tensor<float> *tensor3d_z);

}  // namespace cudaTransformer