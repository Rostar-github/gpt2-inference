#include "src/kernels/kernel_utils.h"
#include "src/core/check_cuda.h"
#include "src/kernels/_device_.cuh"

namespace cudaTransformer {

template <typename T>
__global__ void setValKernel(Tensor3D_Kernel<T> tensor, T val) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockDim.z * blockIdx.z + threadIdx.z;

    if (row < tensor.col_size && col < tensor.row_size && batch < tensor.batch_size)
        setValTensor(tensor, batch, row, col, val);
        // tensor.data[tensor.row_size * row + col] = val;
}

template <typename T>
void setVal(Tensor<T> *tensor, T val) {

    const int block_size = 32;

    Tensor3D_Kernel<T> _tensor(*tensor);

    int dim_x = _tensor.col_size;
    int dim_y = _tensor.row_size;
    int dim_z = _tensor.batch_size;

    dim3 block(block_size, block_size, 1);
    dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1, (dim_z - 1) / block.z + 1);
    setValKernel<T><<<grid, block>>>(_tensor, val);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void setVal<float>(Tensor<float> *tensor, float val);

}