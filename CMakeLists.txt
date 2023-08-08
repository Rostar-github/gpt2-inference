cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
Project(cudaTransformer LANGUAGES CXX C CUDA)

# find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "--ptxas-options=-v" CACHE STRING "nvcc flags" FORCE)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

set(SOURCE_DIR .)
set(3RD_PARTY ${SOURCE_DIR}/3rdparty)

set(CUDA_INCLUDE_PATH /usr/local/cuda-11.0/include)
set(CUDA_LIBRARY /usr/local/cuda-11.0/lib64)

set(CNPY_INCLUDE ${3RD_PARTY}/cnpy/include)
set(CNPY_LIBRARY ${3RD_PARTY}/cnpy/lib)
# include_directories(.)
include_directories(${SOURCE_DIR})

include_directories(${CUDA_INCLUDE_PATH})
link_directories(${CUDA_LIBRARY})

include_directories(${CNPY_INCLUDE})
link_directories(${CNPY_LIBRARY})

add_subdirectory(src)
add_subdirectory(test)

