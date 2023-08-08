#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {


template <typename T>
void softmax(const Tensor<T>* input, Tensor<T>* output, int num_class);

}