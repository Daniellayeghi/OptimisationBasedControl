
#ifndef OPTCONTROL_MUJOCO_GENERIC_CONTROL_H
#define OPTCONTROL_MUJOCO_GENERIC_CONTROL_H

#include "mujoco.h"
#include "../parameters/simulation_params.h"
#include "../src/utilities/generic_utils.h"

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


    virtual void compute_state_value_vec()
    {
        T& underlying = static_cast<T&>(*this); underlying.compute_state_value_vec();
    }


public:
    std::vector<CtrlVector> m_u_traj_new;
    std::vector<CtrlVector> m_u_traj_cp;
    std::vector<CtrlVector> m_u_traj;
    std::vector<StateVector> m_x_traj_new;
    std::vector<StateVector> m_x_traj;
    CtrlVector cached_control;
    std::vector<GenericUtils::FastPair<StateVector, double>> m_state_value;
};

#endif //OPTCONTROL_MUJOCO_GENERIC_CONTROL_H
