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
include test/CMakeFiles/model_test.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/model_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/model_test.dir/flags.make

test/CMakeFiles/model_test.dir/model_test.cpp.o: test/CMakeFiles/model_test.dir/flags.make
test/CMakeFiles/model_test.dir/model_test.cpp.o: ../test/model_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/model_test.dir/model_test.cpp.o"
	cd /wanglina/cuda/cuda_example/build/test && /usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/model_test.dir/model_test.cpp.o -c /wanglina/cuda/cuda_example/test/model_test.cpp

test/CMakeFiles/model_test.dir/model_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/model_test.dir/model_test.cpp.i"
	cd /wanglina/cuda/cuda_example/build/test && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /wanglina/cuda/cuda_example/test/model_test.cpp > CMakeFiles/model_test.dir/model_test.cpp.i

test/CMakeFiles/model_test.dir/model_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/model_test.dir/model_test.cpp.s"
	cd /wanglina/cuda/cuda_example/build/test && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /wanglina/cuda/cuda_example/test/model_test.cpp -o CMakeFiles/model_test.dir/model_test.cpp.s

test/CMakeFiles/model_test.dir/model_test.cpp.o.requires:

.PHONY : test/CMakeFiles/model_test.dir/model_test.cpp.o.requires

test/CMakeFiles/model_test.dir/model_test.cpp.o.provides: test/CMakeFiles/model_test.dir/model_test.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/model_test.dir/build.make test/CMakeFiles/model_test.dir/model_test.cpp.o.provides.build
.PHONY : test/CMakeFiles/model_test.dir/model_test.cpp.o.provides

test/CMakeFiles/model_test.dir/model_test.cpp.o.provides.build: test/CMakeFiles/model_test.dir/model_test.cpp.o


# Object files for target model_test
model_test_OBJECTS = \
"CMakeFiles/model_test.dir/model_test.cpp.o"

# External object files for target model_test
model_test_EXTERNAL_OBJECTS =

test/CMakeFiles/model_test.dir/cmake_device_link.o: test/CMakeFiles/model_test.dir/model_test.cpp.o
test/CMakeFiles/model_test.dir/cmake_device_link.o: test/CMakeFiles/model_test.dir/build.make
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/models/gpt/libgpt.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/layers/libembedding.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/layers/libdecoder.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/layers/libmh_attention.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/layers/libself_attention.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libtensor_map.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/layers/libfeed_forward.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/layers/liblm_head.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libtensor.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libmem_pool.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/kernels/libkernels.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libtensor.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libmem_pool.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/kernels/libkernels.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libmem_ops.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: src/core/libcheck_cuda.a
test/CMakeFiles/model_test.dir/cmake_device_link.o: test/CMakeFiles/model_test.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/model_test.dir/cmake_device_link.o"
	cd /wanglina/cuda/cuda_example/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/model_test.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/model_test.dir/build: test/CMakeFiles/model_test.dir/cmake_device_link.o

.PHONY : test/CMakeFiles/model_test.dir/build

# Object files for target model_test
model_test_OBJECTS = \
"CMakeFiles/model_test.dir/model_test.cpp.o"

# External object files for target model_test
model_test_EXTERNAL_OBJECTS =

test/model_test: test/CMakeFiles/model_test.dir/model_test.cpp.o
test/model_test: test/CMakeFiles/model_test.dir/build.make
test/model_test: src/models/gpt/libgpt.a
test/model_test: src/layers/libembedding.a
test/model_test: src/layers/libdecoder.a
test/model_test: src/layers/libmh_attention.a
test/model_test: src/layers/libself_attention.a
test/model_test: src/core/libtensor_map.a
test/model_test: src/layers/libfeed_forward.a
test/model_test: src/layers/liblm_head.a
test/model_test: src/core/libtensor.a
test/model_test: src/core/libmem_pool.a
test/model_test: src/kernels/libkernels.a
test/model_test: src/core/libtensor.a
test/model_test: src/core/libmem_pool.a
test/model_test: src/kernels/libkernels.a
test/model_test: src/core/libmem_ops.a
test/model_test: src/core/libcheck_cuda.a
test/model_test: test/CMakeFiles/model_test.dir/cmake_device_link.o
test/model_test: test/CMakeFiles/model_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable model_test"
	cd /wanglina/cuda/cuda_example/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/model_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/model_test.dir/build: test/model_test

.PHONY : test/CMakeFiles/model_test.dir/build

test/CMakeFiles/model_test.dir/requires: test/CMakeFiles/model_test.dir/model_test.cpp.o.requires

.PHONY : test/CMakeFiles/model_test.dir/requires

test/CMakeFiles/model_test.dir/clean:
	cd /wanglina/cuda/cuda_example/build/test && $(CMAKE_COMMAND) -P CMakeFiles/model_test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/model_test.dir/clean

test/CMakeFiles/model_test.dir/depend:
	cd /wanglina/cuda/cuda_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /wanglina/cuda/cuda_example /wanglina/cuda/cuda_example/test /wanglina/cuda/cuda_example/build /wanglina/cuda/cuda_example/build/test /wanglina/cuda/cuda_example/build/test/CMakeFiles/model_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/model_test.dir/depend
