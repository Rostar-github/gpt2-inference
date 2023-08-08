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
include src/core/CMakeFiles/check_cuda.dir/depend.make

# Include the progress variables for this target.
include src/core/CMakeFiles/check_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include src/core/CMakeFiles/check_cuda.dir/flags.make

src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o: src/core/CMakeFiles/check_cuda.dir/flags.make
src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o: ../src/core/check_cuda.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o"
	cd /wanglina/cuda/cuda_example/build/src/core && /usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/check_cuda.dir/check_cuda.cpp.o -c /wanglina/cuda/cuda_example/src/core/check_cuda.cpp

src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/check_cuda.dir/check_cuda.cpp.i"
	cd /wanglina/cuda/cuda_example/build/src/core && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /wanglina/cuda/cuda_example/src/core/check_cuda.cpp > CMakeFiles/check_cuda.dir/check_cuda.cpp.i

src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/check_cuda.dir/check_cuda.cpp.s"
	cd /wanglina/cuda/cuda_example/build/src/core && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /wanglina/cuda/cuda_example/src/core/check_cuda.cpp -o CMakeFiles/check_cuda.dir/check_cuda.cpp.s

src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.requires:

.PHONY : src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.requires

src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.provides: src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.requires
	$(MAKE) -f src/core/CMakeFiles/check_cuda.dir/build.make src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.provides.build
.PHONY : src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.provides

src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.provides.build: src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o


# Object files for target check_cuda
check_cuda_OBJECTS = \
"CMakeFiles/check_cuda.dir/check_cuda.cpp.o"

# External object files for target check_cuda
check_cuda_EXTERNAL_OBJECTS =

src/core/libcheck_cuda.a: src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o
src/core/libcheck_cuda.a: src/core/CMakeFiles/check_cuda.dir/build.make
src/core/libcheck_cuda.a: src/core/CMakeFiles/check_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcheck_cuda.a"
	cd /wanglina/cuda/cuda_example/build/src/core && $(CMAKE_COMMAND) -P CMakeFiles/check_cuda.dir/cmake_clean_target.cmake
	cd /wanglina/cuda/cuda_example/build/src/core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/check_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/core/CMakeFiles/check_cuda.dir/build: src/core/libcheck_cuda.a

.PHONY : src/core/CMakeFiles/check_cuda.dir/build

src/core/CMakeFiles/check_cuda.dir/requires: src/core/CMakeFiles/check_cuda.dir/check_cuda.cpp.o.requires

.PHONY : src/core/CMakeFiles/check_cuda.dir/requires

src/core/CMakeFiles/check_cuda.dir/clean:
	cd /wanglina/cuda/cuda_example/build/src/core && $(CMAKE_COMMAND) -P CMakeFiles/check_cuda.dir/cmake_clean.cmake
.PHONY : src/core/CMakeFiles/check_cuda.dir/clean

src/core/CMakeFiles/check_cuda.dir/depend:
	cd /wanglina/cuda/cuda_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /wanglina/cuda/cuda_example /wanglina/cuda/cuda_example/src/core /wanglina/cuda/cuda_example/build /wanglina/cuda/cuda_example/build/src/core /wanglina/cuda/cuda_example/build/src/core/CMakeFiles/check_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/core/CMakeFiles/check_cuda.dir/depend
