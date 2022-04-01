
set(SOURCE_FILES
        hand_manip_cp.cpp
        ilqr_cartpole.cpp
        ilqr_control.cpp
        ilqr_double_integrator.cpp
        ilqr_finger.cpp
        ilqr_franka.cpp
        par_mppi_ilqr_hand_manip.cpp
        ilqr_planar.cpp
        mppi_ilqr_planar_obstacle.cpp
        mppi_ilqr_cartpole.cpp
        mppi_ilqr_finger.cpp
        mppi_ilqr_planar_push.cpp
        mppi_ilqr_planar_push_2d.cpp
        mppi_ilqr_sh_hand.cpp
        mppi_ilqr_sh_hand_par.cpp
        mppi_ilqr_double_cartpole.cpp
        mppi_ilqr_cartpole_par.cpp
        mppi_ilqr_finger_par.cpp
        mppi_ilqr_planar_obstacle_par.cpp
        mppi_ilqr_planar_push_2d_par.cpp#
        mppi_ilqr_walker_par.cpp
#        fic_ilqr_mppi_finger.cpp
#        fic_ilqr_extended_cartpole.cpp
#        fic_ilqr_planar_3_link.cpp
#        fic_ilqr_cartpole.cpp
        )

include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${ZeroMQ_INCLUDE_DIR})

foreach(source ${SOURCE_FILES})
    string(REPLACE ".cpp" "" BIN_NAME ${source})
    add_executable(${BIN_NAME} ${source})
    target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
    #Standard libraries for GL
    target_link_libraries(${BIN_NAME} glut)
    target_link_libraries(${BIN_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
    # Additional libraries from mujoco package
    target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)
    # Additional libraries imgui package
    target_link_libraries(${BIN_NAME} imgui)
    # link fic
    target_link_libraries(${BIN_NAME} fic_lib)
    # General implementation packages
    target_link_libraries(${BIN_NAME} control_lib)
    target_link_libraries(${BIN_NAME} util_lib)
    #target_link_libraries(${BIN_NAME} autodiff::autodiff)
    #target_link_libraries(${BIN_NAME} "${TORCH_LIBRARIES}")
    target_link_libraries(${BIN_NAME} /usr/local/lib/libzmq.so.5)
    target_link_libraries(${BIN_NAME} OpenMP::OpenMP_CXX)

    #    target_link_libraries(${BIN_NAME} ${ZMQPP})
    ## add the include directory to our compile directives
endforeach(source ${SOURCE_FILES})