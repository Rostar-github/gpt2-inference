
#include <cuda_runtime.h>

#ifndef CHECK_CUDA_H
#define CHECK_CUDA_H

namespace cudaTransformer {

#define CHECK(err) (checkError(err, __FILE__, __LINE__))
#define ERRORLOG(err_msg) (errorLog(err_msg, __FILE__, __LINE__))

void checkError(cudaError_t err, const char* file, int line);
void errorLog(const char* err_msg,  const char* file, int line);

}  // namespace cudaTransformer

#endif
