#include "src/kernels/concate_heads.h"
#include "src/kernels/_device_.cuh"
#include "src/core/check_cuda.h"

namespace cudaTransformer {

template <typename T>
__global__ void concateHeadsKernel(Tensor3D_Kernel<T> output, T* addr, size_t size, int dim) {

    int stride = dim;

    int n_samples = output.col_size;

    T* data_addr = addr + blockIdx.y * size;

    int block_col = threadIdx.y;

    int batch = blockIdx.z;

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < n_samples)
        setValTensor(output, batch, row, col, data_addr[batch * stride * n_samples + row * stride + block_col]);
    
}

template <typename T>
void concateHeads(Tensor<T>* output, const std::vector<Tensor<T>*> &tensors) {
    Tensor3D_Kernel<T> _output(*output);

    int dim_x = _output.col_size;
    int dim_y = _output.row_size;
    int dim_z = _output.batch_size;

    size_t tensor_size = tensors[0]->size();
    T* addr = tensors[0]->data;
    int tensor_dim = 64;

    dim3 block(1024 / tensor_dim, tensor_dim, 1);
    dim3 grid((dim_x - 1) / block.x + 1, dim_y / block.y, dim_z / block.z);
    concateHeadsKernel<T><<<grid, block>>>(_output, addr, tensor_size, tensor_dim);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void concateHeads<float>(Tensor<float>* output, const std::vector<Tensor<float>*> &tensors);

}