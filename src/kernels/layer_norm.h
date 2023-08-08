#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {


template <typename T>
void layerNorm(const Tensor<T>* input, 
                Tensor<T>* output, 
                const Tensor<T>* alpha, 
                const Tensor<T>* beta, 
                T epsilon);

}