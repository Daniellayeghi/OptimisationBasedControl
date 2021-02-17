
#ifndef OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
#define OPTCONTROL_MUJOCO_INTERNAL_TYPES_H

#include "mjmodel.h"
#include "eigen3/Eigen/Core"
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace InternalTypes
{
    template<typename T, int state_size, int ctrl_size>
    struct SystemTypes
    {
        using ctrl_vector  = Eigen::Matrix<T, ctrl_size, 1>;
        using state_vector = Eigen::Matrix<T, state_size, 1>;
    };


    template<typename T, int state_size, int ctrl_size>
    struct CostTypes
    {
        using ctrl_cost  = Eigen::Matrix<T, ctrl_size, ctrl_size>;
        using state_cost = Eigen::Matrix<T, state_size, state_size>;
    };
}


namespace AutoDiffTypes
{
    using namespace InternalTypes;
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
