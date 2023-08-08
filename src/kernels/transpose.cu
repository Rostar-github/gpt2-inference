
#include "src/kernels/transpose.h"
#include "src/core/check_cuda.h"
#include "src/kernels/_device_.cuh"

namespace cudaTransformer {

template <typename T>
__global__ void transposeKernel(const Tensor3D_Kernel<T> input, Tensor3D_Kernel<T> output) {

    const int block_size = 32;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockIdx.z;

    __shared__ T shmem[block_size][block_size];

    if (batch < output.batch_size && row < output.col_size && col < output.row_size) {
        
        shmem[threadIdx.y][threadIdx.x] = getValTensor(input, batch, col, row);

        __syncthreads();

        setValTensor(output, batch, row, col, shmem[threadIdx.y][threadIdx.x]);
    }
}

template <typename T>
void transpose(const Tensor<T> *input, Tensor<T> *output) {
    // assert
    assert(input->shape[2] == output->shape[1] && input->shape[1] == output->shape[2]);
    // change data format from tensor to tensor under kernel
    Tensor3D_Kernel<T> _input(*input);
    Tensor3D_Kernel<T> _output(*output);

    int dim_x = _output.col_size;
    int dim_y = _output.row_size;
    int dim_z = _output.batch_size;

    const int block_size = 32;
    
    dim3 block(block_size, block_size, 1);
    dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1, dim_z / block.z);
    transposeKernel<T><<<grid, block>>>(_input, _output);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void transpose<float>(const Tensor<float> *input, Tensor<float> *output);

}



// template <typename T>
// __global__ void transpose2DKernel(const Tensor2D_Kernel<T> tensor2d_x, Tensor2D_Kernel<T> tensor2d_y) {
//     int row = blockDim.x * blockIdx.x + threadIdx.x;
//     int col = blockDim.y * blockIdx.y + threadIdx.y;

//     int tid_row = threadIdx.x;
//     int tid_col = threadIdx.y;

//     __shared__ T shmem[block_size_2d][block_size_2d];

//     shmem[tid_col][tid_row] = tensor2d_x.data[tensor2d_x.row_size * col + row];  
//     __syncthreads();

//     tensor2d_y.data[tensor2d_y.row_size * row + col] = shmem[tid_col][tid_row];
// }


// template <typename T>
// void transpose2D(const Tensor<T> *tensor2d_x, Tensor<T> *tensor2d_y) {
//     // assert
//     assert(tensor2d_x->shape[1] == tensor2d_y->shape[0] && tensor2d_x->shape[0] == tensor2d_y->shape[1]);
//     // change data format from tensor to tensor under kernel
//     Tensor2D_Kernel<T> kernel_tensor2d_x(*tensor2d_x);
//     Tensor2D_Kernel<T> kernel_tensor2d_y(*tensor2d_y);

//     int dim_x = kernel_tensor2d_y.col_size;
//     int dim_y = kernel_tensor2d_y.row_size;
    
//     dim3 block(block_size_2d, block_size_2d);
//     dim3 grid(dim_x / block.x, dim_y / block.y);
//     transpose2DKernel<T><<<grid, block>>>(kernel_tensor2d_x, kernel_tensor2d_y);
//     CHECK(cudaGetLastError());
// }
// template void transpose2D<float>(const Tensor<float> *tensor2d_x, Tensor<float> *tensor2d_y);