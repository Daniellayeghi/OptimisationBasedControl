#ifndef OPTCONTROL_MUJOCO_GENERIC_UTILS_H
#define OPTCONTROL_MUJOCO_GENERIC_UTILS_H

namespace GenericUtils
{
    template<typename T1, typename T2>
    struct FastPair
    {
        T1 first;
        T2 second;
    };
}

#endif //OPTCONTROL_MUJOCO_GENERIC_UTILS_H
