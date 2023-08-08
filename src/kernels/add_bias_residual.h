#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {


template <typename T>
void addBiasResidual(Tensor<T>* input, const Tensor<T>* bias, const Tensor<T>* residual);

}