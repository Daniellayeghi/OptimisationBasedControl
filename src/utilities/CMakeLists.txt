SET(UTILITY_HEADER_LIST finite_diff.h)
SET(UTILITY_SOURCE_LIST finite_diff.cpp)
add_library(util_lib ${UTILITY_SOURCE_LIST} ${UTILITY_HEADER_LIST})
target_link_libraries(util_lib PUBLIC autodiff::autodiff)