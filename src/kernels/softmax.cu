#include "src/kernels/softmax.h"
#include "src/core/check_cuda.h"
#include "src/kernels/_device_.cuh"
#include <float.h>
#include <cmath>
#include <climits>

namespace cudaTransformer {

template <typename T>
__device__ T warpReduceSum(T val)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);  // instruction shift: thread_id t get the val of i + offset
    }
    return val;
}

template <typename T>
__device__ T warpReduceMax(T val)
{
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));  
    }
    return val;
}

// a block handle a row data (size < 1024)
template <typename T>
__global__ void softmaxTinyKernel(const Tensor3D_Kernel<T> input, Tensor3D_Kernel<T> output, int num_class) {

    const int block_size = 1024;

    int row = blockIdx.x;

    int col = threadIdx.y;

    int batch = blockIdx.z;

    T val = (T)0;

    __shared__ T buffer_max[block_size];
    __shared__ T buffer_sum[block_size];

    __shared__ T max_val;
    __shared__ T sum;

    if (threadIdx.y < num_class) {
        val = getValTensor(input, batch, row, col);      
        buffer_max[threadIdx.y] = val;
        buffer_sum[threadIdx.y] = val;
    } else {
        buffer_max[threadIdx.y] = FLT_MIN;
        buffer_sum[threadIdx.y] = (T)0;
    }   

    __syncthreads();

    // block reduce max
    if (num_class > warpSize) {
        for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {
            if (threadIdx.y < offset) {
                buffer_max[threadIdx.y] = fmaxf(buffer_max[threadIdx.y], buffer_max[threadIdx.y + offset]);
            }
            __syncthreads();
        }
    }

    T reg_max_val = buffer_max[threadIdx.y]; // move to register for warpReduce
    
    reg_max_val = warpReduceMax(reg_max_val);

    if (threadIdx.y == 0) max_val = reg_max_val;


    __syncthreads();

    if (threadIdx.y < num_class){
        buffer_sum[threadIdx.y] = expf(buffer_sum[threadIdx.y] - max_val);
    }
       
    __syncthreads();

    // block reduce sum
    if (num_class > warpSize) {
        for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {
            if (threadIdx.y < offset) {
                buffer_sum[threadIdx.y] += buffer_sum[threadIdx.y + offset];
            }
            __syncthreads();
        }
    }
    

    T reg_sum = buffer_sum[threadIdx.y];

    reg_sum = warpReduceSum(reg_sum);

    if (threadIdx.y == 0) sum = reg_sum;

    __syncthreads();


    if (col < num_class)
        // setValTensor(output, batch, row, col, expf(val - max_val) / sum);
        setValTensor(output, batch, row, col, __fdividef(expf(val - max_val), sum));

}

template <typename T>
__global__ void softmaxKernel(const Tensor3D_Kernel<T> input, Tensor3D_Kernel<T> output, int num_class) {

    const int block_size = 1024;

    int row = blockIdx.x;

    int col = threadIdx.y;

    int batch = blockIdx.z;

    __shared__ T max_val;
    __shared__ T sum;

    max_val = FLT_MIN;

    sum = 0;

    for (int _offset = 0; _offset < num_class; _offset += block_size) {

        T val = (T)0;

        __shared__ T buffer_max[block_size];

        if (_offset + col < num_class) {

            val = getValTensor(input, batch, row, _offset + col);   

            buffer_max[threadIdx.y] = val;
        } else {

            buffer_max[threadIdx.y] = FLT_MIN;
        } 

        __syncthreads();

        for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {
            
            if (threadIdx.y < offset) {
                buffer_max[threadIdx.y] = fmaxf(buffer_max[threadIdx.y], buffer_max[threadIdx.y + offset]);
            }
            __syncthreads();
        }

        T reg_max_val = buffer_max[threadIdx.y];

        reg_max_val = warpReduceMax(reg_max_val);

        if (threadIdx.y == 0) max_val = fmaxf(max_val, reg_max_val);

        __syncthreads();
    }   

    for (int offset = 0; offset < num_class; offset += block_size) {

        if (offset + col < num_class) {

            T val = getValTensor(input, batch, row, offset + col);

            setValTensor(output, batch, row, offset + col, expf(val - max_val));

        }
    }

    for (int _offset = 0; _offset < num_class; _offset += block_size) {

        T val = (T)0;

        __shared__ T buffer_sum[block_size];

        if (_offset + col < num_class) {

            val = getValTensor(output, batch, row, _offset + col);      

            buffer_sum[threadIdx.y] = val;
        } else {

            buffer_sum[threadIdx.y] = (T)0;
        } 

        __syncthreads();

        for (int offset = block_size >> 1; offset >= warpSize; offset >>= 1) {
            if (threadIdx.y < offset) {
                
                buffer_sum[threadIdx.y] += buffer_sum[threadIdx.y + offset];
            }
            __syncthreads();
        }

        T reg_sum = buffer_sum[threadIdx.y];

        reg_sum = warpReduceSum(reg_sum);

        if (threadIdx.y == 0) sum += reg_sum;

        __syncthreads();
    }   

    for (int offset = 0; offset < num_class; offset += block_size) {

        if (offset + col < num_class) {

            T val = getValTensor(output, batch, row, offset + col);

            setValTensor(output, batch, row, offset + col, __fdividef(val, sum));
        }
    }

}

template <typename T>
void softmax(const Tensor<T>* input, Tensor<T>* output, int num_class) {
    Tensor3D_Kernel<T> _input(*input);
    Tensor3D_Kernel<T> _output(*output);

    int dim_x = _output.col_size;
    int dim_y = _output.row_size;
    int dim_z = _output.batch_size;

    const int block_size = 1024;

    if (num_class <= 1024) {

        dim3 block(1, block_size, 1);

        dim3 grid(dim_x / block.x , (dim_y - 1) / block.y + 1, dim_z / block.z);

        softmaxTinyKernel<T><<<grid, block>>>(_input, _output, num_class);
    } else {
        
        dim3 block(1, block_size, 1);

        dim3 grid(dim_x / block.x , 1, dim_z / block.z);

        softmaxKernel<T><<<grid, block>>>(_input, _output, num_class);
    }
        
    
    CHECK(cudaGetLastError());

    // cudaDeviceSynchronize();
}
template void softmax<float>(const Tensor<float>* input, Tensor<float>* output, int num_class);

}
