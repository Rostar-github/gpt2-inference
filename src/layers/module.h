#pragma once

#include <vector>

namespace cudaTransformer {

template<typename T>
class Module {

public:
    Module(bool free_buf_after_forward): free_buf_after_forward(free_buf_after_forward){};

protected:
    bool free_buf_after_forward;
    virtual void allocateBuffer(size_t batch_size, size_t n_samples) = 0;
    virtual void freeBuffer() = 0;
    virtual ~Module() = default;
    
};


}