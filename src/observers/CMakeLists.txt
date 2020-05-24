SET(OBSERVER_HEADER_LIST sensors.h)
SET(OBSERVER_SOURCE_LIST sensors.cpp)
add_library(observer_lib ${OBSERVER_SOURCE_LIST} ${OBSERVER_HEADER_LIST})
target_link_libraries(observer_lib Eigen3::Eigen util_lib)