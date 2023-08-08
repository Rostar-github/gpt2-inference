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
include src/layers/CMakeFiles/mh.dir/depend.make

# Include the progress variables for this target.
include src/layers/CMakeFiles/mh.dir/progress.make

# Include the compile flags for this target's objects.
include src/layers/CMakeFiles/mh.dir/flags.make

src/layers/CMakeFiles/mh.dir/mlp.cpp.o: src/layers/CMakeFiles/mh.dir/flags.make
src/layers/CMakeFiles/mh.dir/mlp.cpp.o: ../src/layers/mlp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/layers/CMakeFiles/mh.dir/mlp.cpp.o"
	cd /wanglina/cuda/cuda_example/build/src/layers && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mh.dir/mlp.cpp.o -c /wanglina/cuda/cuda_example/src/layers/mlp.cpp

src/layers/CMakeFiles/mh.dir/mlp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mh.dir/mlp.cpp.i"
	cd /wanglina/cuda/cuda_example/build/src/layers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /wanglina/cuda/cuda_example/src/layers/mlp.cpp > CMakeFiles/mh.dir/mlp.cpp.i

src/layers/CMakeFiles/mh.dir/mlp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mh.dir/mlp.cpp.s"
	cd /wanglina/cuda/cuda_example/build/src/layers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /wanglina/cuda/cuda_example/src/layers/mlp.cpp -o CMakeFiles/mh.dir/mlp.cpp.s

src/layers/CMakeFiles/mh.dir/mlp.cpp.o.requires:

.PHONY : src/layers/CMakeFiles/mh.dir/mlp.cpp.o.requires

src/layers/CMakeFiles/mh.dir/mlp.cpp.o.provides: src/layers/CMakeFiles/mh.dir/mlp.cpp.o.requires
	$(MAKE) -f src/layers/CMakeFiles/mh.dir/build.make src/layers/CMakeFiles/mh.dir/mlp.cpp.o.provides.build
.PHONY : src/layers/CMakeFiles/mh.dir/mlp.cpp.o.provides

src/layers/CMakeFiles/mh.dir/mlp.cpp.o.provides.build: src/layers/CMakeFiles/mh.dir/mlp.cpp.o


# Object files for target mh
mh_OBJECTS = \
"CMakeFiles/mh.dir/mlp.cpp.o"

# External object files for target mh
mh_EXTERNAL_OBJECTS =

src/layers/libmh.a: src/layers/CMakeFiles/mh.dir/mlp.cpp.o
src/layers/libmh.a: src/layers/CMakeFiles/mh.dir/build.make
src/layers/libmh.a: src/layers/CMakeFiles/mh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libmh.a"
	cd /wanglina/cuda/cuda_example/build/src/layers && $(CMAKE_COMMAND) -P CMakeFiles/mh.dir/cmake_clean_target.cmake
	cd /wanglina/cuda/cuda_example/build/src/layers && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mh.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/layers/CMakeFiles/mh.dir/build: src/layers/libmh.a

.PHONY : src/layers/CMakeFiles/mh.dir/build

src/layers/CMakeFiles/mh.dir/requires: src/layers/CMakeFiles/mh.dir/mlp.cpp.o.requires

.PHONY : src/layers/CMakeFiles/mh.dir/requires

src/layers/CMakeFiles/mh.dir/clean:
	cd /wanglina/cuda/cuda_example/build/src/layers && $(CMAKE_COMMAND) -P CMakeFiles/mh.dir/cmake_clean.cmake
.PHONY : src/layers/CMakeFiles/mh.dir/clean

src/layers/CMakeFiles/mh.dir/depend:
	cd /wanglina/cuda/cuda_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /wanglina/cuda/cuda_example /wanglina/cuda/cuda_example/src/layers /wanglina/cuda/cuda_example/build /wanglina/cuda/cuda_example/build/src/layers /wanglina/cuda/cuda_example/build/src/layers/CMakeFiles/mh.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/layers/CMakeFiles/mh.dir/depend

