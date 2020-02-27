SET(CONTROL_HEADER_LIST controller.h ../utilities/finite_diff.h cost_function.h)
SET(CONTROL_SOURCE_LIST controller.cpp ../utilities/finite_diff.cpp cost_function.cpp)
add_library(control_lib ${CONTROL_SOURCE_LIST} ${CONTROL_HEADER_LIST})
target_link_libraries(control_lib PUBLIC autodiff::autodiff util_lib)