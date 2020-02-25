
#ifndef OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
#define OPTCONTROL_MUJOCO_INTERNAL_TYPES_H

#include "mjmodel.h"
#include "eigen3/Eigen/Core"

namespace InternalTypes
{
    using Mat9x1 = Eigen::Matrix<mjtNum, 9, 1>;
    using Mat9x2 = Eigen::Matrix<mjtNum, 9, 2>;
    using Mat9x3 = Eigen::Matrix<mjtNum, 9, 3>;
}

#endif //OPTCONTROL_MUJOCO_INTERNAL_TYPES_H
