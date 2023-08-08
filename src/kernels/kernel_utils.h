#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
void setVal(Tensor<T>* tensor, T val);


}