SET(CONTROL_HEADER_LIST controller.h cost_function.h ilqr.h mppi_ddp.h ../../third_party/FIC/fic.h generic_control.h par_mppi_ddp.h)
SET(CONTROL_SOURCE_LIST controller.cpp cost_function.cpp ilqr.cpp mppi_ddp.cpp par_mppi_ddp.cpp)
add_library(control_lib ${CONTROL_SOURCE_LIST} ${CONTROL_HEADER_LIST})
target_link_libraries(control_lib Eigen3::Eigen util_lib "${TORCH_LIBRARIES}")