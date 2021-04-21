#ifndef OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
#define OPTCONTROL_MUJOCO_MUJOCO_UTILS_H

namespace MujocoUtils
{
    //Randomly position obstacles
    void populate_obstacles(const int start_id,const int end_id, const std::array<double, 6> &bounds, const mjModel *model);
}
#endif //OPTCONTROL_MUJOCO_MUJOCO_UTILS_H
