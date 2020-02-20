SET(CONTROL_HEADER_LIST controller.h)
SET(CONTROL_SOURCE_LIST controller.cpp ../utilities/finite_diff.cpp ../utilities/finite_diff.h)
add_library(control_lib ${CONTROL_SOURCE_LIST} ${CONTROL_HEADER_LIST})
target_link_libraries(control_lib PUBLIC autodiff::autodiff util_lib)