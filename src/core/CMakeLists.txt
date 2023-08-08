

add_library(mem_pool mem_pool.cpp)
target_link_libraries(mem_pool tensor mem_ops -lcudart)

add_library(check_cuda check_cuda.cpp)
target_link_libraries(check_cuda -lcudart)

add_library(mem_ops mem_ops.cpp)
target_link_libraries(mem_ops  check_cuda -lcudart)

add_library(tensor tensor.cpp)
target_link_libraries(tensor mem_ops check_cuda kernels mem_pool -lcudart -lcnpy)

add_library(tensor_map tensor_map.cpp)
target_link_libraries(tensor_map  tensor -lcudart -lcnpy)
