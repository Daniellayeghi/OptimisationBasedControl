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
CMAKE_SOURCE_DIR = /home/daniel/Repos/OptimisationBasedControl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daniel/Repos/OptimisationBasedControl/build-release

# Include any dependencies generated for this target.
include src/utilities/CMakeFiles/util_lib.dir/depend.make

# Include the progress variables for this target.
include src/utilities/CMakeFiles/util_lib.dir/progress.make

# Include the compile flags for this target's objects.
include src/utilities/CMakeFiles/util_lib.dir/flags.make

src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o: src/utilities/CMakeFiles/util_lib.dir/flags.make
src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o: ../src/utilities/finite_diff.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/Repos/OptimisationBasedControl/build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o"
	cd /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/util_lib.dir/finite_diff.cpp.o -c /home/daniel/Repos/OptimisationBasedControl/src/utilities/finite_diff.cpp

src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/util_lib.dir/finite_diff.cpp.i"
	cd /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/Repos/OptimisationBasedControl/src/utilities/finite_diff.cpp > CMakeFiles/util_lib.dir/finite_diff.cpp.i

src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/util_lib.dir/finite_diff.cpp.s"
	cd /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/Repos/OptimisationBasedControl/src/utilities/finite_diff.cpp -o CMakeFiles/util_lib.dir/finite_diff.cpp.s

src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.requires:

.PHONY : src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.requires

src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.provides: src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.requires
	$(MAKE) -f src/utilities/CMakeFiles/util_lib.dir/build.make src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.provides.build
.PHONY : src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.provides

src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.provides.build: src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o


# Object files for target util_lib
util_lib_OBJECTS = \
"CMakeFiles/util_lib.dir/finite_diff.cpp.o"

# External object files for target util_lib
util_lib_EXTERNAL_OBJECTS =

src/utilities/libutil_lib.a: src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o
src/utilities/libutil_lib.a: src/utilities/CMakeFiles/util_lib.dir/build.make
src/utilities/libutil_lib.a: src/utilities/CMakeFiles/util_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/daniel/Repos/OptimisationBasedControl/build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libutil_lib.a"
	cd /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities && $(CMAKE_COMMAND) -P CMakeFiles/util_lib.dir/cmake_clean_target.cmake
	cd /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/util_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/utilities/CMakeFiles/util_lib.dir/build: src/utilities/libutil_lib.a

.PHONY : src/utilities/CMakeFiles/util_lib.dir/build

src/utilities/CMakeFiles/util_lib.dir/requires: src/utilities/CMakeFiles/util_lib.dir/finite_diff.cpp.o.requires

.PHONY : src/utilities/CMakeFiles/util_lib.dir/requires

src/utilities/CMakeFiles/util_lib.dir/clean:
	cd /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities && $(CMAKE_COMMAND) -P CMakeFiles/util_lib.dir/cmake_clean.cmake
.PHONY : src/utilities/CMakeFiles/util_lib.dir/clean

src/utilities/CMakeFiles/util_lib.dir/depend:
	cd /home/daniel/Repos/OptimisationBasedControl/build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daniel/Repos/OptimisationBasedControl /home/daniel/Repos/OptimisationBasedControl/src/utilities /home/daniel/Repos/OptimisationBasedControl/build-release /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities /home/daniel/Repos/OptimisationBasedControl/build-release/src/utilities/CMakeFiles/util_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/utilities/CMakeFiles/util_lib.dir/depend

