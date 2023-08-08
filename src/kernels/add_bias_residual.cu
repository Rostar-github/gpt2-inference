#include "src/kernels/add_bias_residual.h"
#include "src/core/check_cuda.h"
#include "src/kernels/_device_.cuh"

namespace cudaTransformer {

template <typename T>
__global__ void addBiasResidualKernel(Tensor3D_Kernel<T> input, 
                                        const Tensor2D_Kernel<T> bias, const Tensor3D_Kernel<T> residual) {
    int row = blockIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockIdx.z;

    if (col < input.row_size) {
        T input_val = getValTensor(input, batch, row, col);
        T res_val = input_val + bias.data[col] + getValTensor(residual, batch, row, col);
        setValTensor(input, batch, row, col, res_val);
    }

    
}


template <typename T>
void addBiasResidual(Tensor<T>* input, const Tensor<T>* bias, const Tensor<T>* residual) {
                    
    Tensor3D_Kernel<T> _input(*input);
    Tensor3D_Kernel<T> _residual(*residual);
    Tensor2D_Kernel<T> _bias(*bias);

    int dim_x = _input.col_size;
    int dim_y = _input.row_size;
    int dim_z = _input.batch_size;

    const int block_size = 1024;
    // support hidden_size = 768, 1024

    dim3 block(1, block_size, 1);
    dim3 grid(dim_x / block.x, (dim_y - 1) / block.y + 1, dim_z / block.z);
    addBiasResidualKernel<T><<<grid, block>>>(_input, _bias, _residual);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void addBiasResidual<float>(Tensor<float>* input, const Tensor<float>* bias, const Tensor<float>* residual);


}