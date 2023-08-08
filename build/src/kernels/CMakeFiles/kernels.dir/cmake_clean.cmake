file(REMOVE_RECURSE
  "CMakeFiles/kernels.dir/gemm.cu.o"
  "CMakeFiles/kernels.dir/transpose.cu.o"
  "CMakeFiles/kernels.dir/input_encoder.cu.o"
  "CMakeFiles/kernels.dir/kernel_utils.cu.o"
  "CMakeFiles/kernels.dir/qkv_add_bias.cu.o"
  "CMakeFiles/kernels.dir/qk_scale_gemm.cu.o"
  "CMakeFiles/kernels.dir/softmax.cu.o"
  "CMakeFiles/kernels.dir/layer_norm.cu.o"
  "CMakeFiles/kernels.dir/concate_heads.cu.o"
  "CMakeFiles/kernels.dir/add_bias_residual_ln.cu.o"
  "CMakeFiles/kernels.dir/add_bias_gelu.cu.o"
  "CMakeFiles/kernels.dir/add_bias_residual.cu.o"
  "CMakeFiles/kernels.dir/gready_search.cu.o"
  "libkernels.pdb"
  "libkernels.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/kernels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
