/**
 * allocator.cc
 * 内存配置器，负责在Tensor中分配内存，作为Tensor的内置默认分配器
 * 基本内存操作由memory_ops支持
*/
#include "src/core/mem_pool.h"
#include "src/core/mem_ops.h"
#include "src/core/check_cuda.h"
#include <iostream>

namespace cudaTransformer {

template <typename T>
MemoryPool<T>::MemoryPool(size_t _capacity, Memtype _memtype):
capacity(_capacity), memtype(_memtype) {
    // setup memory pool GPU
    new_slot();
}

template <typename T>
void MemoryPool<T>::new_slot() {
    if (memtype == GPU) {
        Slot slot(0, capacity - 1);

        deviceMalloc(slot.buffer, capacity);

        slots.push_back(slot);

    } else if (memtype == CPU) {
        Slot slot(0, capacity - 1);

        hostMalloc(slot.buffer, capacity);

        slots.push_back(slot);

    } else if (memtype == UNI_MEM) {
        Slot slot(0, capacity - 1);

        deviceManageMem(slot.buffer, capacity);

        slots.push_back(slot);

    } else {
        Slot slot(0, capacity - 1);

        hostPinnedMem(slot.buffer, capacity);

        slots.push_back(slot);
    }
}

template <typename T>
T* MemoryPool<T>::dynamic_allocate(size_t size) {
    if (size > capacity) ERRORLOG("Size Error: max size of Tensor(dynamic) is larger than memory slot.");
    std::cout << "slots size: " << slots.size() << std::endl;
    // First-fit algorithm
    for (size_t i = 0; i < slots.size(); i++) { 
        if (slots[i].right <= slots[i].left + size) {
            std::cout << "slot num:" << i + 1 << ", next location: " << -1 << std::endl;
            std::cout << "-------------------" << std::endl;
            continue;
        } else {
            size_t next = slots[i].right - size;
            slots[i].right = next;
            std::cout << "slot num:" << i + 1 << ", next location: " << next << " addr: "<< &slots[i].buffer[next+1] << std::endl;
            return &slots[i].buffer[next + 1];
        }
    }
    new_slot();
    Slot slot = slots.back();
    size_t next = slot.right - size;
    slot.right = next;
    std::cout << "slot num:" << slots.size() << ", next location: " << next << " addr: "<< &slot.buffer[next+1] << std::endl;
    return &slot.buffer[next + 1];
}

template <typename T>
T* MemoryPool<T>::static_allocate(size_t size) {
    if (size > capacity) ERRORLOG("Size Error: max size of Tensor(static) is larger than memory slot.");
    for (size_t i = 0; i < slots.size(); i++) {
        size_t next = slots[i].left + size;
        if (next >= slots[i].right) {
            continue;
        } else {
            T* pointer = &slots[i].buffer[slots[i].left];
            slots[i].left = next;
            return pointer;
        }
    }
    new_slot();
    Slot slot = slots.back();
    size_t next = slot.left + size;
    T* pointer = &slot.buffer[slot.left];
    slot.left = next;
    return pointer;
}

template <typename T>
void MemoryPool<T>::dynamic_deallocate(T* buffer, size_t size) {
    for (size_t i = 0; i < slots.size(); i++) {
        std::cout << "addr_slot:   " << &slots[i].buffer[slots[i].right + 1] << std::endl;
        std::cout << "addr_buffer: " << buffer << std::endl;
        if (slots[i].right + 1 < capacity && &slots[i].buffer[slots[i].right + 1] == buffer) {
            slots[i].right += size;
            std::cout << "deallocate position: " << slots[i].right << std::endl;
            return;
        }
    }
    ERRORLOG("Error dynamic deallocate.");
}

template <typename T>
void MemoryPool<T>::static_deallocate(T* buffer, size_t size) {
    for (size_t i = 0; i < slots.size(); i++) {
        if (slots[i].left - size >= 0 && &slots[i].buffer[slots[i].left - size] == buffer) {
            slots[i].left -= size;
        }
    }
    ERRORLOG("Error static deallocate.");
}

template <typename T>
MemoryPool<T>::~MemoryPool() {
    
    for (Slot &slot : slots) {

        if (memtype == GPU) 

            deviceFree(slot.buffer);

        else if (memtype == CPU)

            hostFree(slot.buffer);

        else if (memtype == UNI_MEM)

            deviceFree(slot.buffer);

        else 
            hostPinFree(slot.buffer);
    }
}

template class MemoryPool<float>;

}
