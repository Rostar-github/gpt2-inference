#include "check_cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

namespace cudaTransformer {
    
void checkError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err),
                cudaGetErrorString(err), file, line);
        exit(int(err));
    }
}

void errorLog(const char* err_msg,  const char* file, int line) {
    fprintf(stderr, "%s || in %s at line %d\n", err_msg, file, line);
    exit(-1);
}

}  // namespace cudaTransformer
