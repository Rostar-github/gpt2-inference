#include "src/kernels/add_bias_residual_ln.h"
#include "src/kernels/_device_.cuh"
#include "src/core/check_cuda.h"

namespace cudaTransformer {

template <typename T>
__device__ T warpReduceSum(T val)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);  
    }
    return val;
}

template <typename T>
__global__ void addBiasResidualLNKernel(const Tensor3D_Kernel<T> input, 
                                Tensor3D_Kernel<T> output,
                                const Tensor2D_Kernel<T> bias,
                                const Tensor3D_Kernel<T> residual,
                                const Tensor2D_Kernel<T> alpha,
                                const Tensor2D_Kernel<T> beta,
                                T epsilon) {

    const int block_size = 1024;

    int hidden_size = input.row_size;

    int row = blockIdx.x;

    int col = threadIdx.y;

    int batch = blockIdx.z;

    T val;
    T residual_val;

    __shared__ T buffer_mean_sum[block_size];
    __shared__ T buffer_std_sum[block_size];

    __shared__ T mean;
    __shared__ T std;

    if (threadIdx.y < hidden_size) {
        val = getValTensor(input, batch, row, col);
        residual_val = getValTensor(residual, batch, row, col);
        buffer_mean_sum[threadIdx.y] = val + residual_val + bias.data[threadIdx.y];
        buffer_std_sum[threadIdx.y] = val + residual_val + bias.data[threadIdx.y];
    } else {
        buffer_mean_sum[threadIdx.y] = (T)0;
        buffer_std_sum[threadIdx.y] = (T)0;
    }

    __syncthreads();

    for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {
        if (threadIdx.y < offset) {
            buffer_mean_sum[threadIdx.y] += buffer_mean_sum[threadIdx.y + offset];
        }
        __syncthreads();
    }

    T reg_mean_sum = buffer_mean_sum[threadIdx.y];

    reg_mean_sum = warpReduceSum(reg_mean_sum);

    if (threadIdx.y == 0) mean = __fdividef(reg_mean_sum, hidden_size);

    __syncthreads();

    if (threadIdx.y < hidden_size) {
        buffer_std_sum[threadIdx.y] = powf(buffer_std_sum[threadIdx.y] - mean, 2.0);
    }

    __syncthreads();

    for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {
        if (threadIdx.y < offset) {
            buffer_std_sum[threadIdx.y] += buffer_std_sum[threadIdx.y + offset];
        }
        __syncthreads();
    }

    T reg_std_sum = buffer_std_sum[threadIdx.y];

    reg_std_sum = warpReduceSum(reg_std_sum);

    if (threadIdx.y == 0) std = sqrtf(__fdividef(reg_std_sum, hidden_size));

    __syncthreads();

    // element-wise

    if (threadIdx.y < hidden_size) {

        T x = getValTensor(input, batch, row, col);

        // x = alpha.data[threadIdx.y] * (x - mean) / (std + epsilon) + beta.data[threadIdx.y];

        x = __fdividef(alpha.data[threadIdx.y] * (x - mean), std + epsilon) + beta.data[threadIdx.y];

        setValTensor(output, batch, row, col, x);

    }
    

}

template <typename T>
void addBiasResidualLN(const Tensor<T>* input, 
                        Tensor<T>* output,
                        const Tensor<T>* bias,
                        const Tensor<T>* residual,
                        const Tensor<T>* alpha,
                        const Tensor<T>* beta,
                        T epsilon) {
                    
    Tensor3D_Kernel<T> _input(*input);
    Tensor3D_Kernel<T> _output(*output);
    Tensor3D_Kernel<T> _residual(*residual);

    Tensor2D_Kernel<T> _bias(*bias);
    Tensor2D_Kernel<T> _alpha(*alpha);
    Tensor2D_Kernel<T> _beta(*beta);

    int dim_x = _output.col_size;
    int dim_y = _output.row_size;
    int dim_z = _output.batch_size;

    const int block_size = 1024;
    // support hidden_size = 768, 1024

    dim3 block(1, block_size, 1);
    dim3 grid(dim_x / block.x , (dim_y - 1) / block.y + 1, dim_z / block.z);
    addBiasResidualLNKernel<T><<<grid, block>>>(_input, _output, _bias, _residual, _alpha, _beta, epsilon);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void addBiasResidualLN<float>(const Tensor<float>* input, 
                                        Tensor<float>* output,
                                        const Tensor<float>* bias,
                                        const Tensor<float>* residual,
                                        const Tensor<float>* alpha,
                                        const Tensor<float>* beta,
                                        float epsilon);

}