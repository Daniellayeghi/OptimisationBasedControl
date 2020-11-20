
set(BINARIES
        ilqr_cartpole.cpp
        ilqr_control.cpp
        ilqr_double_integrator.cpp
        ilqr_finger.cpp
        ilqr_franka.cpp
        ilqr_franka_sparse.cpp
        ilqr_pendulum.cpp
        mppi_control_finger.cpp
        mppi_control_cartpole.cpp
        )

include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

foreach(bin_file ${BINARIES})
    string(REPLACE ".cpp" "" BIN_NAME ${bin_file})
    add_executable(${BIN_NAME} ${bin_file})
    target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
    #Standard libraries for GL
    target_link_libraries(${BIN_NAME} glut)
    target_link_libraries(${BIN_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
    # Additional libraries from mujoco package
    target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)
    # Additional libraries imgui_1 package
    target_link_libraries(${BIN_NAME} imgui)
    # General implementation packages
    target_link_libraries(${BIN_NAME} control_lib)
    target_link_libraries(${BIN_NAME} util_lib)
    #target_link_libraries(${BIN_NAME} autodiff::autodiff)
    target_link_libraries(${BIN_NAME} "${TORCH_LIBRARIES}")
endforeach(bin_file ${BINARIES})