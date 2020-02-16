SET(CONTROL_HEADER_LIST controller.h)
SET(CONTROL_SOURCE_LIST controller.cpp)
add_library(control_lib ${CONTROL_SOURCE_LIST} ${CONTROL_HEADER_LIST})
target_link_libraries(control_lib PUBLIC autodiff::autodiff)