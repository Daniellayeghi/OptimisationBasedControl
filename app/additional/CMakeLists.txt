
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(BINARIES simulate.cpp)

include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

foreach(bin_file ${BINARIES})
    string(REPLACE ".cpp" "" BIN_NAME ${bin_file})
    add_executable(${BIN_NAME} ${bin_file})
    target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
    #Standard libraries for GL
    target_link_libraries(${BIN_NAME} glut)
    target_link_libraries(${BIN_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
    # Additional libraries from mujoco package
    target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)
    target_link_libraries(${BIN_NAME} util_lib)
    target_link_libraries(${BIN_NAME} ui_lib)
endforeach(bin_file ${BINARIES})