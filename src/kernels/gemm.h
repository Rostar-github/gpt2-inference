#pragma once

#include "src/core/tensor.h"

namespace cudaTransformer {

// template <typename T>
// void gemm2D(const Tensor<T> *tensor3d_x, const Tensor<T> *tensor2d_y, Tensor<T> *tensor3d_z);

template <typename T>
void gemm(const Tensor<T> *tensor3d_x, const Tensor<T> *tensor3d_y, Tensor<T> *tensor3d_z);

}
