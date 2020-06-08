SET(UTILITY_HEADER_LIST finite_diff.h internal_types.h basic_math.h buffer_utils.h)
SET(UTILITY_SOURCE_LIST finite_diff.cpp  basic_math.cpp)
add_library(util_lib ${UTILITY_SOURCE_LIST} ${UTILITY_HEADER_LIST})
target_link_libraries(util_lib PUBLIC Eigen3::Eigen)
