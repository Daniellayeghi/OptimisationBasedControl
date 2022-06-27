
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(BINARIES simulate.cc)

# Utility library
add_library(uitools STATIC)
target_sources(uitools PRIVATE uitools.h uitools.c)
target_include_directories(uitools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(uitools PUBLIC glfw mujoco::mujoco)

foreach(bin_file ${BINARIES})
    string(REPLACE ".cpp" "" BIN_NAME ${bin_file})
    add_executable(${BIN_NAME} ${bin_file})
    target_link_libraries(${BIN_NAME} mujoco::mujoco)
    target_link_libraries(${BIN_NAME} glfw)
    target_link_libraries(${BIN_NAME} util_lib)
    target_link_libraries(${BIN_NAME} uitools)
endforeach(bin_file ${BINARIES})
