add_library(kernels 
gemm.cu 
transpose.cu 
input_encoder.cu 
kernel_utils.cu
qkv_add_bias.cu
qk_scale_gemm.cu
softmax.cu
layer_norm.cu
concate_heads.cu
add_bias_residual_ln.cu
add_bias_gelu.cu
add_bias_residual.cu
gready_search.cu)
target_link_libraries(kernels  tensor check_cuda -lcudart)
