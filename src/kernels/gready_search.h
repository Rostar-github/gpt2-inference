#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {


template <typename T>
void greadySearch(const Tensor<T>* input, Tensor<T>* output, Tensor<T>* max_val, int n_samples, int vocab_size);

}