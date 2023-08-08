#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {


template <typename T>
void addBiasGeluInPlace(Tensor<T>* input, const Tensor<T>* bias);

}