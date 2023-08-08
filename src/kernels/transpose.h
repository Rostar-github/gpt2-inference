#pragma once

#include <cuda_runtime.h>
#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
void transpose(const Tensor<T> *input, Tensor<T> *output);

}