
# Copyright (c) 2017, Atabak Dehban
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Inspired by http://www.mujoco.org/forum/index.php?threads/cmake-request.3361/#post-3747

cmake_minimum_required(VERSION 3.5)
project(OptControl_MUJOCO)

set(USE_GL 1)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

#add_compile_options(-Wall -Wextra)
find_package(autodiff REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# N.B. This is a temporary flag. It only really applies to Linux, as Mac
# does not need X11.
option(RUN_X11_TESTS "Run tests that require X11" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#Mujoco paths
set(MUJOCO_PY_MJPRO_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED MUJ_KEY_PATH)
    message(STATUS "Addingg definition Mujoco key path defined as:" ${MUJ_KEY_PATH})
    add_definitions(-DMUJ_KEY_PATH="${MUJ_KEY_PATH}")
else()
    message(FATAL_ERROR "Mujoco key path NOT defined")
endif()

OPTION(DEFINE_DEBUG "Build the project using debugging code" OFF)
IF(DEFINE_DEBUG)
    MESSAGE("Adding Debug flag...")
    ADD_DEFINITIONS(-DDEBUG)
    SET(CMAKE_BUILD_TYPE Debug)
    MESSAGE("Build type is " ${CMAKE_BUILD_TYPE})
ENDIF(DEFINE_DEBUG)

include_directories(${MUJOCO_PY_MJPRO_PATH}/include)
link_directories(${MUJOCO_PY_MJPRO_PATH}/libraries)

#Finding main mujoco library
if(${USE_GL})
    file(GLOB LIB_MUJOCO ${MUJOCO_PY_MJPRO_PATH}/libraries/libmujoco[0-9][0-9][0-9].so)
else()
    file(GLOB LIB_MUJOCO ${MUJOCO_PY_MJPRO_PATH}/libraries/libmujoco[0-9][0-9][0-9]nogl.so)
endif()

#Showing mujoco library found
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)