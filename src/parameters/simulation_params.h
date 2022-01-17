
#ifndef OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
#define OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
#include "Eigen/Core"

namespace SimulationParameters
{
    /* Environment Dimensions */
    using SimScalarType = double;

    constexpr const int n_ctrl = 1;
    constexpr const int n_jpos = 2;
    constexpr const int n_jvel = 2;
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
    template<typename T>
    struct RawType {
        using type = typename T::Scalar[T::RowsAtCompileTime * T::ColsAtCompileTime];
        using scalar = typename T::Scalar;
        static constexpr const unsigned int size = sizeof(T);
    };

    /* Threading */
    constexpr const unsigned int n_threads = 5;
}

#endif //OPTCONTROL_MUJOCO_SIMULATION_PARAMS_H
