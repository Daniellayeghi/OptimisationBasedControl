
#ifndef OPTCONTROL_MUJOCO_GENERIC_CONTROL_H
#define OPTCONTROL_MUJOCO_GENERIC_CONTROL_H

#include <mujoco/mujoco.h>
#include "../parameters/simulation_params.h"
#include "../utilities/generic_utils.h"

using namespace SimulationParameters;

template <typename T>
class BaseController
{
public:
    explicit BaseController() = default;

    virtual void control(const mjData* data, bool skip = false)
    {
        T& underlying = static_cast<T&>(*this); underlying.control(data, skip);
    }

public:
    std::vector<CtrlVector> m_u_traj_new;
    std::vector<CtrlVector> m_u_traj_cp;
    std::vector<CtrlVector> m_u_traj;
    std::vector<StateVector> m_x_traj_new;
    std::vector<StateVector> m_x_traj;
    CtrlVector cached_control;
    GenericUtils::FastPair<std::vector<StateVector>, std::vector<double>> m_state_value;
};

#endif //OPTCONTROL_MUJOCO_GENERIC_CONTROL_H
