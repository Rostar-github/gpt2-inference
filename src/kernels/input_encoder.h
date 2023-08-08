#pragma once

#include <cuda_runtime.h>

#include "src/core/tensor.h"

namespace cudaTransformer {

template <typename T>
void oneHotEncoder(const Tensor<T>* tokens, Tensor<T>* onehot);

template <typename T>
void positionEncoder(Tensor<T>* posi_code);

template <typename T>
void embeddingEncoder(const Tensor<T>* onehot, const Tensor<T>* posi_code,
                      const Tensor<T>* token_embed_w,
                      const Tensor<T>* posi_embed_w, Tensor<T>* hidden);

}  // namespace cudaTransformer
