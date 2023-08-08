add_library(gpt gpt.cpp)
target_link_libraries(gpt tensor tensor_map mem_pool embedding decoder lm_head -lcudart)