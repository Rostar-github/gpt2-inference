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
include src/layers/CMakeFiles/decoder.dir/depend.make

# Include the progress variables for this target.
include src/layers/CMakeFiles/decoder.dir/progress.make

# Include the compile flags for this target's objects.
include src/layers/CMakeFiles/decoder.dir/flags.make

src/layers/CMakeFiles/decoder.dir/decoder.cpp.o: src/layers/CMakeFiles/decoder.dir/flags.make
src/layers/CMakeFiles/decoder.dir/decoder.cpp.o: ../src/layers/decoder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/layers/CMakeFiles/decoder.dir/decoder.cpp.o"
	cd /wanglina/cuda/cuda_example/build/src/layers && /usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/decoder.dir/decoder.cpp.o -c /wanglina/cuda/cuda_example/src/layers/decoder.cpp

src/layers/CMakeFiles/decoder.dir/decoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/decoder.dir/decoder.cpp.i"
	cd /wanglina/cuda/cuda_example/build/src/layers && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /wanglina/cuda/cuda_example/src/layers/decoder.cpp > CMakeFiles/decoder.dir/decoder.cpp.i

src/layers/CMakeFiles/decoder.dir/decoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/decoder.dir/decoder.cpp.s"
	cd /wanglina/cuda/cuda_example/build/src/layers && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /wanglina/cuda/cuda_example/src/layers/decoder.cpp -o CMakeFiles/decoder.dir/decoder.cpp.s

src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.requires:

.PHONY : src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.requires

src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.provides: src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.requires
	$(MAKE) -f src/layers/CMakeFiles/decoder.dir/build.make src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.provides.build
.PHONY : src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.provides

src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.provides.build: src/layers/CMakeFiles/decoder.dir/decoder.cpp.o


# Object files for target decoder
decoder_OBJECTS = \
"CMakeFiles/decoder.dir/decoder.cpp.o"

# External object files for target decoder
decoder_EXTERNAL_OBJECTS =

src/layers/libdecoder.a: src/layers/CMakeFiles/decoder.dir/decoder.cpp.o
src/layers/libdecoder.a: src/layers/CMakeFiles/decoder.dir/build.make
src/layers/libdecoder.a: src/layers/CMakeFiles/decoder.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/wanglina/cuda/cuda_example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libdecoder.a"
	cd /wanglina/cuda/cuda_example/build/src/layers && $(CMAKE_COMMAND) -P CMakeFiles/decoder.dir/cmake_clean_target.cmake
	cd /wanglina/cuda/cuda_example/build/src/layers && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/decoder.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/layers/CMakeFiles/decoder.dir/build: src/layers/libdecoder.a

.PHONY : src/layers/CMakeFiles/decoder.dir/build

src/layers/CMakeFiles/decoder.dir/requires: src/layers/CMakeFiles/decoder.dir/decoder.cpp.o.requires

.PHONY : src/layers/CMakeFiles/decoder.dir/requires

src/layers/CMakeFiles/decoder.dir/clean:
	cd /wanglina/cuda/cuda_example/build/src/layers && $(CMAKE_COMMAND) -P CMakeFiles/decoder.dir/cmake_clean.cmake
.PHONY : src/layers/CMakeFiles/decoder.dir/clean

src/layers/CMakeFiles/decoder.dir/depend:
	cd /wanglina/cuda/cuda_example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /wanglina/cuda/cuda_example /wanglina/cuda/cuda_example/src/layers /wanglina/cuda/cuda_example/build /wanglina/cuda/cuda_example/build/src/layers /wanglina/cuda/cuda_example/build/src/layers/CMakeFiles/decoder.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/layers/CMakeFiles/decoder.dir/depend

