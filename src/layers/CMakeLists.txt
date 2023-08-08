# add_library(module module.h)
# target_link_libraries(module  tensor check_cuda -lcudart)

add_library(embedding embedding.cpp)
target_link_libraries(embedding  tensor tensor_map kernels mem_pool -lcudart)

add_library(self_attention self_attention.cpp)
target_link_libraries(self_attention tensor tensor_map kernels mem_pool -lcudart)

add_library(feed_forward feed_forward.cpp)
target_link_libraries(feed_forward mem_pool -lcudart)

add_library(decoder decoder.cpp)
target_link_libraries(decoder mh_attention feed_forward mem_pool -lcudart)

add_library(lm_head lm_head.cpp)
target_link_libraries(lm_head mem_pool -lcudart)


add_library(mh_attention mh_attention.cpp)
target_link_libraries(mh_attention self_attention  mem_pool -lcudart)

add_library(ln_norm ln_norm.cpp)
target_link_libraries(ln_norm mem_pool -lcudart)


