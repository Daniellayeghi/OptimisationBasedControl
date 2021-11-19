FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.8.0
)

FetchContent_MakeAvailable(googletest)

#add_executable(solver_tests finite_diff_tests.cpp)
##Standard libraries for GL
## Additional libraries from mujoco package
#include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
#target_link_libraries(solver_tests ${LIB_MUJOCO})
#target_link_libraries(solver_tests gtest_main gtest util_lib Eigen3::Eigen)
#target_link_libraries(solver_tests libglew.so libglfw.so libglewegl.so libglewosmesa.so)
#target_link_libraries(solver_tests ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
#target_link_libraries(solver_tests glut)


add_executable(open_mp_tests open_mp_tests.cpp)
target_link_libraries(open_mp_tests gtest_main gtest util_lib Eigen3::Eigen)
target_link_libraries(open_mp_tests OpenMP::OpenMP_CXX)
