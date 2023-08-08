#include "src/kernels/add_bias_gelu.h"
#include "src/kernels/_device_.cuh"
#include "src/core/check_cuda.h"
#include <cmath>

namespace cudaTransformer {

template <typename T>
__global__ void addBiasGeluInPlaceKernel(Tensor3D_Kernel<T> input, const Tensor2D_Kernel<T> bias) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockIdx.z;

    if (row < input.col_size && col < input.row_size) {
        T val = getValTensor(input, batch, row, col);
        val += bias.data[col];
        val = 0.5f * val * (1.0f + tanf(sqrtf(__fdividef(2.0f, M_PI)) * (val + 0.044715f * powf(val, 3.0f))));
        setValTensor(input, batch, row, col, val);
    }
}


template <typename T>
void addBiasGeluInPlace(Tensor<T>* input, const Tensor<T>* bias) {
                    
    Tensor3D_Kernel<T> _input(*input);
    Tensor2D_Kernel<T> _bias(*bias);

    int dim_x = _input.col_size;
    int dim_y = _input.row_size;
    int dim_z = _input.batch_size;

    const int block_size = 32;
    // support hidden_size = 768, 1024

    dim3 block(block_size, block_size, 1);
    dim3 grid((dim_x - 1) / block.x + 1, (dim_y - 1) / block.y + 1, dim_z / block.z);
    addBiasGeluInPlaceKernel<T><<<grid, block>>>(_input, _bias);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void addBiasGeluInPlace<float>(Tensor<float>* input, const Tensor<float>* bias);


}