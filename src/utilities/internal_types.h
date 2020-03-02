
#ifndef OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
#define OPTCONTROL_MUJOCO_INTERNAL_TYPES_H

#include "mjmodel.h"
#include "eigen3/Eigen/Core"
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace InternalTypes
{
    using Mat9x1 = Eigen::Matrix<mjtNum, 9, 1>;
    using Mat9x2 = Eigen::Matrix<mjtNum, 9, 2>;
    using Mat9x3 = Eigen::Matrix<mjtNum, 9, 3>;
}


namespace AutoDiffTypes
{
    using inner_derivative_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using inner_active_scalar   = Eigen::AutoDiffScalar<inner_derivative_type>;
    using outer_derivative_type = Eigen::Matrix<inner_active_scalar, Eigen::Dynamic, 1>;
    using outer_active_scalar   = Eigen::AutoDiffScalar<outer_derivative_type>;
    using AVector               = Eigen::Matrix<outer_active_scalar, Eigen::Dynamic, 1>;
}

#endif //OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
