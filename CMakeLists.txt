cmake_minimum_required(VERSION 3.14)
project(OptControl_MUJOCO)

set(USE_GL 1)
set(BUILD_BINDINGS 0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include(FetchContent)

add_compile_options(-Wall -Wextra)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(mujoco REQUIRED)
#find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZeroMQ REQUIRED)

# N.B. This is a temporary flag. It only really applies to Linux, as Mac
# does not need X11.
option(RUN_X11_TESTS "Run tests that require X11" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wattributes -mavx -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


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

#Showing mujoco library found
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
add_subdirectory(third_party)

if(${BUILD_BINDINGS})
    add_subdirectory(bindings)
endif()
