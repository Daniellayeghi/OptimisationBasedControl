SET(CONTROL_HEADER_LIST ${PROJECT_SOURCE_DIR}/controller/controller.h)
SET(CONTROL_SOURCE_LIST ${PROJECT_SOURCE_DIR}/controller/controller.cpp)
add_library(control_lib ${CONTROL_SOURCE_LIST} ${CONTROL_HEADER_LIST})