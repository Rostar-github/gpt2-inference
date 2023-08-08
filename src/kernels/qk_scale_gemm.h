#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
void qkScaleGemm(const Tensor<T> *q, const Tensor<T> *k, Tensor<T> *qk, const T scale);

}