# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /wanglina/cuda/cuda_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /wanglina/cuda/cuda_example/build

# Include any dependencies generated for this target.
include test/CMakeFiles/layer_test.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/layer_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/layer_test.dir/flags.make

test/CMakeFiles/layer_test.dir/layer_test.cpp.o: test/CMakeFiles/layer_test.dir/flags.make
test/CMakeFiles/layer_test.dir/layer_test.cpp.o: ../test/layer_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/layer_test.dir/layer_test.cpp.o"
	cd /wanglina/cuda/cuda_example/build/test && /usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/layer_test.dir/layer_test.cpp.o -c /wanglina/cuda/cuda_example/test/layer_test.cpp

test/CMakeFiles/layer_test.dir/layer_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/layer_test.dir/layer_test.cpp.i"
	cd /wanglina/cuda/cuda_example/build/test && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /wanglina/cuda/cuda_example/test/layer_test.cpp > CMakeFiles/layer_test.dir/layer_test.cpp.i

test/CMakeFiles/layer_test.dir/layer_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/layer_test.dir/layer_test.cpp.s"
	cd /wanglina/cuda/cuda_example/build/test && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /wanglina/cuda/cuda_example/test/layer_test.cpp -o CMakeFiles/layer_test.dir/layer_test.cpp.s

test/CMakeFiles/layer_test.dir/layer_test.cpp.o.requires:

.PHONY : test/CMakeFiles/layer_test.dir/layer_test.cpp.o.requires

test/CMakeFiles/layer_test.dir/layer_test.cpp.o.provides: test/CMakeFiles/layer_test.dir/layer_test.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/layer_test.dir/build.make test/CMakeFiles/layer_test.dir/layer_test.cpp.o.provides.build
.PHONY : test/CMakeFiles/layer_test.dir/layer_test.cpp.o.provides

test/CMakeFiles/layer_test.dir/layer_test.cpp.o.provides.build: test/CMakeFiles/layer_test.dir/layer_test.cpp.o


# Object files for target layer_test
layer_test_OBJECTS = \
"CMakeFiles/layer_test.dir/layer_test.cpp.o"

# External object files for target layer_test
layer_test_EXTERNAL_OBJECTS =

test/CMakeFiles/layer_test.dir/cmake_device_link.o: test/CMakeFiles/layer_test.dir/layer_test.cpp.o
test/CMakeFiles/layer_test.dir/cmake_device_link.o: test/CMakeFiles/layer_test.dir/build.make
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/libembedding.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/libdecoder.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/liblm_head.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/libln_norm.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/libmh_attention.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/libself_attention.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libtensor_map.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/layers/libfeed_forward.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libtensor.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/kernels/libkernels.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libmem_pool.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libtensor.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/kernels/libkernels.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libmem_pool.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libmem_ops.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: src/core/libcheck_cuda.a
test/CMakeFiles/layer_test.dir/cmake_device_link.o: test/CMakeFiles/layer_test.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/layer_test.dir/cmake_device_link.o"
	cd /wanglina/cuda/cuda_example/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/layer_test.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/layer_test.dir/build: test/CMakeFiles/layer_test.dir/cmake_device_link.o

.PHONY : test/CMakeFiles/layer_test.dir/build

# Object files for target layer_test
layer_test_OBJECTS = \
"CMakeFiles/layer_test.dir/layer_test.cpp.o"

# External object files for target layer_test
layer_test_EXTERNAL_OBJECTS =

test/layer_test: test/CMakeFiles/layer_test.dir/layer_test.cpp.o
test/layer_test: test/CMakeFiles/layer_test.dir/build.make
test/layer_test: src/layers/libembedding.a
test/layer_test: src/layers/libdecoder.a
test/layer_test: src/layers/liblm_head.a
test/layer_test: src/layers/libln_norm.a
test/layer_test: src/layers/libmh_attention.a
test/layer_test: src/layers/libself_attention.a
test/layer_test: src/core/libtensor_map.a
test/layer_test: src/layers/libfeed_forward.a
test/layer_test: src/core/libtensor.a
test/layer_test: src/kernels/libkernels.a
test/layer_test: src/core/libmem_pool.a
test/layer_test: src/core/libtensor.a
test/layer_test: src/kernels/libkernels.a
test/layer_test: src/core/libmem_pool.a
test/layer_test: src/core/libmem_ops.a
test/layer_test: src/core/libcheck_cuda.a
test/layer_test: test/CMakeFiles/layer_test.dir/cmake_device_link.o
test/layer_test: test/CMakeFiles/layer_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable layer_test"
	cd /wanglina/cuda/cuda_example/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/layer_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/layer_test.dir/build: test/layer_test

.PHONY : test/CMakeFiles/layer_test.dir/build

test/CMakeFiles/layer_test.dir/requires: test/CMakeFiles/layer_test.dir/layer_test.cpp.o.requires

.PHONY : test/CMakeFiles/layer_test.dir/requires

test/CMakeFiles/layer_test.dir/clean:
	cd /wanglina/cuda/cuda_example/build/test && $(CMAKE_COMMAND) -P CMakeFiles/layer_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/layer_test.dir/clean

test/CMakeFiles/layer_test.dir/depend:
	cd /wanglina/cuda/cuda_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /wanglina/cuda/cuda_example /wanglina/cuda/cuda_example/test /wanglina/cuda/cuda_example/build /wanglina/cuda/cuda_example/build/test /wanglina/cuda/cuda_example/build/test/CMakeFiles/layer_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/layer_test.dir/depend
