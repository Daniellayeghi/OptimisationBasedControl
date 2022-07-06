FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.8.0
)

FetchContent_MakeAvailable(googletest)

add_executable(solver_tests opt_tests.cpp)
# Additional libraries from mujoco package
target_link_libraries(solver_tests mujoco::mujoco)
target_link_libraries(solver_tests glfw)
target_link_libraries(solver_tests gtest_main gtest util_lib control_lib Eigen3::Eigen)

add_executable(open_mp_tests open_mp_tests.cpp)
target_link_libraries(open_mp_tests PUBLIC gtest_main gtest util_lib Eigen3::Eigen mujoco::mujoco)
target_link_libraries(open_mp_tests PUBLIC OpenMP::OpenMP_CXX)
