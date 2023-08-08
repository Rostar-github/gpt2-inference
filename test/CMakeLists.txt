

add_executable(kernel_test kernel_test.cpp)
target_link_libraries(kernel_test tensor kernels mem_pool)

add_executable(layer_test layer_test.cpp)
target_link_libraries(layer_test embedding decoder lm_head ln_norm)

add_executable(model_test model_test.cpp)
target_link_libraries(model_test gpt)