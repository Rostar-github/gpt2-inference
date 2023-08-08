#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {


template <typename T>
void addBiasResidualLN(const Tensor<T>* input, 
                        Tensor<T>* output,
                        const Tensor<T>* bias,
                        const Tensor<T>* residual,
                        const Tensor<T>* alpha,
                        const Tensor<T>* beta,
                        T epsilon);

}