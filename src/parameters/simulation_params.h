
#ifndef OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
#define OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
#include "Eigen/Core"

namespace SimulationParameters
{
    /* Environment Dimensions */
    using SimScalarType = double;

    constexpr const int n_ctrl = 20;
    constexpr const int n_jpos = 25;
    constexpr const int n_jvel = 25;
    constexpr const int state_size = n_jpos + n_jvel;

    /* Matrices Used */
    using CtrlVector  = Eigen::Matrix<SimScalarType, n_ctrl, 1>;
    using CtrlMatrix  = Eigen::Matrix<SimScalarType, n_ctrl, n_ctrl>;
    using PosVector   = Eigen::Matrix<SimScalarType, n_jpos, 1>;
    using VelVector   = Eigen::Matrix<SimScalarType, n_jvel, 1>;
    using StateVector = Eigen::Matrix<SimScalarType, state_size, 1>;
    using StateMatrix = Eigen::Matrix<SimScalarType, state_size, state_size>;
    using CtrlStateMatrix = Eigen::Matrix<SimScalarType, n_ctrl, state_size>;
    using StateCtrlMatrix = Eigen::Matrix<SimScalarType, state_size, n_ctrl>;

    /* Raw Types Used */
    template<typename T_Eigen>
    struct RawType {
        using type = typename T_Eigen::Scalar[T_Eigen::RowsAtCompileTime * T_Eigen::ColsAtCompileTime];
        using scalar = typename T_Eigen::Scalar;
        static constexpr const unsigned int size = sizeof(T_Eigen);
    };

    /* Threading */
    constexpr const unsigned int n_threads = 13;
}

#endif //OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
