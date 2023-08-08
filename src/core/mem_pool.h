#include "src/core/type.h"
#include <vector>

#ifndef MEM_POOL_H
#define MEM_POOL_H

namespace cudaTransformer {

template <typename T>
class MemoryPool {

private:

    struct Slot {
        size_t left;
        size_t right;
        T* buffer;
        Slot(size_t left, size_t right):
        left(left), right(right), buffer(nullptr){}
    };

    std::vector<Slot> slots;

    void new_slot();

public:

    size_t capacity;
    Memtype memtype;

    MemoryPool(size_t _capacity, Memtype _memtype);

    T* dynamic_allocate(size_t size);

    T* static_allocate(size_t size);

    void dynamic_deallocate(T* buffer, size_t size);

    void static_deallocate(T* buffer, size_t size);

    ~MemoryPool();

};

}

#endif