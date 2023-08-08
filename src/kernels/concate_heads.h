#pragma once

#include "src/core/tensor.h"
#include <vector>

namespace cudaTransformer {


template <typename T>
void concateHeads(Tensor<T>* output, const std::vector<Tensor<T>*> &tensors);

}