# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/mytest/test11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/mytest/test11/build

# Include any dependencies generated for this target.
include CMakeFiles/opencv_dec.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencv_dec.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv_dec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv_dec.dir/flags.make

CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o: CMakeFiles/opencv_dec.dir/flags.make
CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o: ../src/opencv_dec.cpp
CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o: CMakeFiles/opencv_dec.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/mytest/test11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o"
	aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o -MF CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o.d -o CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o -c /workspace/mytest/test11/src/opencv_dec.cpp

CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.i"
	aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/mytest/test11/src/opencv_dec.cpp > CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.i

CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.s"
	aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/mytest/test11/src/opencv_dec.cpp -o CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.s

# Object files for target opencv_dec
opencv_dec_OBJECTS = \
"CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o"

# External object files for target opencv_dec
opencv_dec_EXTERNAL_OBJECTS =

bin/opencv_dec: CMakeFiles/opencv_dec.dir/src/opencv_dec.cpp.o
bin/opencv_dec: CMakeFiles/opencv_dec.dir/build.make
bin/opencv_dec: CMakeFiles/opencv_dec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/mytest/test11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/opencv_dec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_dec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv_dec.dir/build: bin/opencv_dec
.PHONY : CMakeFiles/opencv_dec.dir/build

CMakeFiles/opencv_dec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv_dec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv_dec.dir/clean

CMakeFiles/opencv_dec.dir/depend:
	cd /workspace/mytest/test11/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/mytest/test11 /workspace/mytest/test11 /workspace/mytest/test11/build /workspace/mytest/test11/build /workspace/mytest/test11/build/CMakeFiles/opencv_dec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencv_dec.dir/depend

