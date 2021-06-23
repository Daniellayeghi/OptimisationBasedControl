
#ifndef OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
#define OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
#include "Eigen/Core"


namespace SimulationParameters
{
    /* Environment Dimensions */
    constexpr const int n_ctrl = 3;
    constexpr const int n_jpos = 3;
    constexpr const int n_jvel = 3;
    constexpr const int state_size = n_jpos + n_jvel;

    /* Matrices Used */
    using CtrlVector  = Eigen::Matrix<double, n_ctrl, 1>;
    using CtrlMatrix  = Eigen::Matrix<double, n_ctrl, n_ctrl>;
    using StateVector = Eigen::Matrix<double, state_size, 1>;
    using StateMatrix = Eigen::Matrix<double, state_size, state_size>;
    using CtrlStateMatrix = Eigen::Matrix<double, n_ctrl, state_size>;
    using StateCtrlMatrix = Eigen::Matrix<double, state_size, n_ctrl>;

}


#endif //OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
