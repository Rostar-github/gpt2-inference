#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
void qkvAddBias(Tensor<T>* q, Tensor<T>* k, Tensor<T>* v, 
                const Tensor<T>* q_bias, const Tensor<T>* k_bias, const Tensor<T>* v_bias);


}