#include "src/kernels/gready_search.h"
#include "src/core/check_cuda.h"
#include "src/kernels/_device_.cuh"
#include <float.h>

namespace cudaTransformer {

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

template <typename T>
__device__ T warpReduceMax(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template <typename T>
__global__ void findMaxKernel(const Tensor3D_Kernel<T> input, Tensor2D_Kernel<T> max_val, int n_samples, int vocab_size) {

    const int block_size = 1024;

    int row = blockIdx.x;

    // int col = blockDim.y * blockIdx.y + threadIdx.y;

    int batch = blockIdx.z;
    
    int stride = blockDim.y;

    if (row == n_samples - 1) {

        __shared__ T buffer[block_size];

        T val = FLT_MIN;

        for (int i = threadIdx.y; i < vocab_size; i += stride) {

            val = fmaxf(val, getValTensor(input, batch, row, i));
        }

        buffer[threadIdx.y] = val;

        __syncthreads();

        for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {

            if (threadIdx.y < offset) {

                buffer[threadIdx.y] = fmaxf(buffer[threadIdx.y], buffer[threadIdx.y + offset]); 
            }
            __syncthreads();
        }

        T reg_max = buffer[threadIdx.y];

        reg_max = warpReduceMax(reg_max);

        if (threadIdx.y == 0) {

            max_val.data[batch] = reg_max;

        } 
    }

    


    // if (row == n_samples - 1) {

    //     T val = (T)0;

    //     __shared__ T buffer_max[block_size];

    //     if (col < vocab_size) {

    //         val = getValTensor(input, batch, row, col);

    //         buffer_max[threadIdx.y] = val;
    //     } else {

    //         buffer_max[threadIdx.y] = FLT_MIN;
    //     }

    //     __syncthreads();

    //     for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {

    //         if (threadIdx.y < offset) {

    //             buffer_max[threadIdx.y] = fmaxf(buffer_max[threadIdx.y], buffer_max[threadIdx.y + offset]); 
    //         }
    //         __syncthreads();
    //     }

    //     T reg_max = buffer_max[threadIdx.y];

    //     reg_max = warpReduceMax(reg_max);

    //     if (threadIdx.y == 0) {

    //         atomicMaxFloat(max_val.data + batch, reg_max);
    //     }
    // }


}

template <typename T>
__global__ void argMaxKernel(const Tensor3D_Kernel<T> input, Tensor2D_Kernel<T> output, 
                            const Tensor2D_Kernel<T> max_val, int n_samples, int vocab_size) {
                                
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    int row = blockIdx.x;

    int batch = blockIdx.z;

    if (row = n_samples - 1) {

        T max_v = max_val.data[batch];

        if (col < vocab_size) {

            T val = getValTensor(input, batch, row, col);

            if (fabsf(val - max_v) < 1e-6f) {

                output.data[batch] = col;
            }
        }
    }


}

template <typename T>
void greadySearch(const Tensor<T>* input, Tensor<T>* output, Tensor<T>* max_val, int n_samples, int vocab_size) {
    Tensor3D_Kernel<T> _input(*input);
    Tensor2D_Kernel<T> _max_val(*max_val);
    Tensor2D_Kernel<T> _output(*output);

    int dim_x = _input.col_size;
    int dim_y = _input.row_size;
    int dim_z = _input.batch_size;

    const int block_size = 1024;

        
    dim3 block_1(1, block_size, 1);

    dim3 grid_1(dim_x / block_1.x , 1, dim_z / block_1.z);

    findMaxKernel<T><<<grid_1, block_1>>>(_input, _max_val, n_samples, vocab_size);

    dim3 block_2(1, block_size, 1);

    dim3 grid_2(dim_x / block_2.x , (dim_y - 1) / block_2.y + 1, dim_z / block_2.z);

    argMaxKernel<T><<<grid_2, block_2>>>(_input, _output, _max_val, n_samples, vocab_size);    
    
    CHECK(cudaGetLastError());

    // cudaDeviceSynchronize();
}
template void greadySearch<float>(const Tensor<float>* input, Tensor<float>* output, Tensor<float>* max_val, int n_samples, int vocab_size);

}