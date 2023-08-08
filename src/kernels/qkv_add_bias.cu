#include "src/kernels/qkv_add_bias.h"
#include "src/core/check_cuda.h"

namespace cudaTransformer {

template <typename T>
__global__ void qkvAddBiasKernel(Tensor3D_Kernel<T> q, Tensor3D_Kernel<T> k, Tensor3D_Kernel<T> v,
                                Tensor2D_Kernel<T> q_bias, Tensor2D_Kernel<T> k_bias, Tensor2D_Kernel<T> v_bias) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < q.col_size && col < q.row_size && batch < q.batch_size) {
        q.data[q.row_size * q.col_size * batch + q.row_size * row + col] = 
        q_bias.data[q_bias.row_size * row + col];

        k.data[k.row_size * k.col_size * batch + k.row_size * row + col] = 
        k_bias.data[k_bias.row_size * row + col];

        v.data[v.row_size * v.col_size * batch + v.row_size * row + col] = 
        v_bias.data[v_bias.row_size * row + col];
    }
}

template <typename T>
void qkvAddBias(Tensor<T>* q, Tensor<T>* k, Tensor<T>* v, 
                const Tensor<T>* q_bias, const Tensor<T>* k_bias, const Tensor<T>* v_bias) {
    // change data format from tensor to tensor under kernel
    Tensor3D_Kernel<T> _q(*q);
    Tensor3D_Kernel<T> _k(*k);
    Tensor3D_Kernel<T> _v(*v);
    Tensor2D_Kernel<T> _q_bias(*q_bias);
    Tensor2D_Kernel<T> _k_bias(*k_bias);
    Tensor2D_Kernel<T> _v_bias(*v_bias);

    int dim_x = _q.col_size;
    int dim_y = _q.row_size;
    int dim_z = _q.batch_size;
    
    dim3 block(1, 64, 1);
    dim3 grid(dim_x / block.x, (dim_y - 1) / block.y + 1, dim_z / block.z);
    qkvAddBiasKernel<T><<<grid, block>>>(_q, _k, _v, _q_bias, _k_bias, _v_bias);
    CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
}
template void qkvAddBias<float>(Tensor<float>* q, Tensor<float>* k, Tensor<float>* v, 
                                const Tensor<float>* q_bias, const Tensor<float>* k_bias, const Tensor<float>* v_bias);


}