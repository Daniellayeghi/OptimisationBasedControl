
#ifndef OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
#define OPTCONTROL_MUJOCO_INTERNAL_TYPES_H

#include "mjmodel.h"
#include "eigen3/Eigen/Core"
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace InternalTypes
{
    using Mat3x3 = Eigen::Matrix<mjtNum, 3, 3>;
    using Mat2x1 = Eigen::Matrix<mjtNum, 2, 1>;
    using Mat1x2 = Eigen::Matrix<mjtNum, 2, 1>;
    using Mat2x2 = Eigen::Matrix<mjtNum, 2, 2>;
    using Mat2x4 = Eigen::Matrix<mjtNum, 2, 4>;
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
    using Mat4x2 = Eigen::Matrix<mjtNum, 4, 2>;
    using Mat4x4 = Eigen::Matrix<mjtNum, 4, 4>;
    using Mat4x1 = Eigen::Matrix<mjtNum, 4, 1>;
    using Mat4x6 = Eigen::Matrix<mjtNum, 4, 6>;
}


namespace AutoDiffTypes
{
    using inner_derivative_type4x1 = Eigen::Matrix<double, 8, 1>;
    using inner_active_scalar4x1   = Eigen::AutoDiffScalar<inner_derivative_type4x1>;
    using outer_derivative_type4x1 = Eigen::Matrix<inner_active_scalar4x1, 8, 1>;
    using outer_active_scalar4x1   = Eigen::AutoDiffScalar<outer_derivative_type4x1>;
    using AVector4x1               = Eigen::Matrix<outer_active_scalar4x1, 4, 1>;

    using inner_derivative_type2x1 = Eigen::Matrix<double, 2, 1>;
    using inner_active_scalar2x1   = Eigen::AutoDiffScalar<inner_derivative_type2x1>;
    using outer_derivative_type2x1 = Eigen::Matrix<inner_active_scalar2x1, 2, 1>;
    using outer_active_scalar2x1   = Eigen::AutoDiffScalar<outer_derivative_type2x1>;
    using AVector2x1               = Eigen::Matrix<outer_active_scalar2x1, 2, 1>;
}

#endif //OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
