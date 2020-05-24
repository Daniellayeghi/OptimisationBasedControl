
#ifndef OPTCONTROL_MUJOCO_BASIC_MATH_H
#define OPTCONTROL_MUJOCO_BASIC_MATH_H

namespace BasicMath
{
    double wrap_to_max(double x, double max);
    double wrap_to_min_max(double x, double min, double max);
}

#endif //OPTCONTROL_MUJOCO_BASIC_MATH_H
