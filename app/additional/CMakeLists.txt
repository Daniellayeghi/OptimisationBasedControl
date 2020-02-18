set(BIN_NAME derivative)
add_executable(${BIN_NAME} ${BIN_NAME}.cpp)
target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
#Standard libraries for GL
target_link_libraries(${BIN_NAME} GL GLU glut)
# Additional libraries from mujoco package
target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)