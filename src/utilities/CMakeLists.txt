SET(UTILITY_HEADER_LIST finite_diff.h buffer_utils.h buffer.h mujoco_utils.h eigen_norm_dist.h eigen_norm_dist_2.h generic_utils.h zmq_utils.h math_utils.h generic_algs.h fast_derivatives.h)
SET(UTILITY_SOURCE_LIST finite_diff.cpp)
add_library(util_lib ${UTILITY_SOURCE_LIST} ${UTILITY_HEADER_LIST})
target_link_libraries(util_lib PUBLIC Eigen3::Eigen mujoco::mujoco)
