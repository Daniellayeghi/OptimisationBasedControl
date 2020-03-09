
#ifndef OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
#define OPTCONTROL_MUJOCO_INTERNAL_TYPES_H

#include "mjmodel.h"
#include "eigen3/Eigen/Core"
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace InternalTypes
{
    using Mat3x3 = Eigen::Matrix<mjtNum, 3, 3>;
    using Mat3x6 = Eigen::Matrix<mjtNum, 3, 6>;
    using Mat3x9 = Eigen::Matrix<mjtNum, 3, 9>;
    using Mat9x1 = Eigen::Matrix<mjtNum, 9, 1>;
    using Mat9x2 = Eigen::Matrix<mjtNum, 9, 2>;
    using Mat9x3 = Eigen::Matrix<mjtNum, 9, 3>;
    using Mat9x9 = Eigen::Matrix<mjtNum, 9, 9>;
    using Mat6x3 = Eigen::Matrix<mjtNum, 6, 3>;
    using Mat6x1 = Eigen::Matrix<mjtNum, 6, 1>;
    using Mat6x6 = Eigen::Matrix<mjtNum, 6, 6>;
    using Mat6x9 = Eigen::Matrix<mjtNum, 6, 9>;
}


namespace AutoDiffTypes
{
    using inner_derivative_type4x1 = Eigen::Matrix<double, 12, 1>;
    using inner_active_scalar4x1   = Eigen::AutoDiffScalar<inner_derivative_type4x1>;
    using outer_derivative_type4x1 = Eigen::Matrix<inner_active_scalar4x1, 12, 1>;
    using outer_active_scalar4x1   = Eigen::AutoDiffScalar<outer_derivative_type4x1>;
    using AVector4x1               = Eigen::Matrix<outer_active_scalar4x1, 6, 1>;

    using inner_derivative_type2x1 = Eigen::Matrix<double, 2, 1>;
    using inner_active_scalar2x1   = Eigen::AutoDiffScalar<inner_derivative_type2x1>;
    using outer_derivative_type2x1 = Eigen::Matrix<inner_active_scalar2x1, 2, 1>;
    using outer_active_scalar2x1   = Eigen::AutoDiffScalar<outer_derivative_type2x1>;
    using AVector2x1               = Eigen::Matrix<outer_active_scalar2x1, 2, 1>;
}

#endif //OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
