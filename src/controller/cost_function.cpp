#include <iostream>
#include "cost_function.h"
#include "../utilities/mujoco_utils.h"


using namespace InternalTypes;

template<int state_size, int ctrl_size>
CostFunction<state_size, ctrl_size>::CostFunction(const StateVector& x_desired,
                                                  const CtrlVector& u_desired,
                                                  const StateMatrix& x_gain,
                                                  const CtrlMatrix& u_gain,
                                                  const CtrlMatrix& u_diff_gain,
                                                  const StateMatrix& x_terminal_gain,
                                                  const mjModel* m)
                                                  :
                                                  m_u_gain(u_gain),
                                                  m_u_diff_gain(u_diff_gain),
                                                  m_x_gain(x_gain),
                                                  m_x_terminal_gain(x_terminal_gain),
                                                  m_u_desired(u_desired),
                                                  m_x_desired(x_desired),
                                                  m_m(m)
{}


template<int state_size, int ctrl_size>
void CostFunction<state_size, ctrl_size>::update_errors(const mjData *d)
{
    MujocoUtils::fill_state_vector(d, m_x, m_m);
    MujocoUtils::fill_ctrl_vector(d, m_u, m_m);
    m_x_error = m_x - m_x_desired;
    m_u_error = m_u;
    m_du_error = m_u - m_u_prev;
    m_u_prev = m_u;
}


template<int state_size, int ctrl_size>
inline void CostFunction<state_size, ctrl_size>::update_errors(const StateVector& state, const CtrlVector& ctrl)
{
//    for(unsigned int row = 0; row < state_size/2; ++row)
//    {
//        state(row, 0) = state(row, 0);
//    }
    m_x_error = state - m_x_desired;
    m_u_error = ctrl;
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::running_cost(const mjData *d)
{
    update_errors(d);
    return (m_x_error.transpose() * m_x_gain * m_x_error)(0, 0) +
           (m_u_error.transpose() * m_u_gain * m_u_error)(0, 0) +
           (m_du_error.transpose() * m_u_diff_gain * m_du_error)(0, 0);
}


template<int state_size, int ctrl_size>
inline mjtNum CostFunction<state_size, ctrl_size>::trajectory_running_cost(const std::vector<StateVector>& x_trajectory,
                                                                           const std::vector<CtrlVector>& u_trajectory)
{
    auto cost = 0.0;
    m_u_prev = u_trajectory.front();
    //Compute running cost
    for(unsigned int row = 0; row < u_trajectory.size(); ++row)
    {
        update_errors(x_trajectory[row], u_trajectory[row]);
        cost += (m_x_error.transpose() * m_x_gain * m_x_error)(0, 0) +
                (m_u_error.transpose() * m_u_gain * m_u_error)(0, 0) +
                (m_du_error.transpose() * m_u_diff_gain * m_du_error)(0, 0);
    }

    //Compute terminal cost
//    for(unsigned int row = 0; row < state_size/2; ++row)
//    {
//        int jid = m_m->dof_jntid[row];
//        if(m_m->jnt_type[jid] == mjJNT_HINGE)
//            x_trajectory.back()(row, 0) = x_trajectory.back()(row, 0);
//    }
    m_x_error = m_x_desired - x_trajectory.back();
    //Running cost + terminal cost
    return cost + (m_x_error.transpose() * m_x_terminal_gain * m_x_error)(0, 0);
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::terminal_cost(const mjData *d)
{
    update_errors(d);
    return (m_x_error.transpose() * m_x_terminal_gain * m_x_error)(0, 0);
}


template<int state_size, int ctrl_size>
StateVector CostFunction<state_size, ctrl_size>::Lf_x(const mjData *d)
{
    update_errors(d);
    return m_x_error.transpose() * (2 * m_x_terminal_gain);
}


template<int state_size, int ctrl_size>
StateMatrix CostFunction<state_size, ctrl_size>::Lf_xx()
{
    return 2 * m_x_terminal_gain;
}


template<int state_size, int ctrl_size>
StateVector CostFunction<state_size, ctrl_size>::L_x(const mjData *d)
{
    update_errors(d);
    return m_x_error.transpose() * (2 * m_x_gain);
}


template<int state_size, int ctrl_size>
StateMatrix CostFunction<state_size, ctrl_size>::L_xx(const mjData *d)
{
    update_errors(d);
    return 2 * m_x_gain;
}


template<int state_size, int ctrl_size>
CtrlVector CostFunction<state_size, ctrl_size>::L_u(const mjData *d)
{
    update_errors(d);
    return (m_u_error.transpose() * (2 * m_u_gain)) * 2;
}


template<int state_size, int ctrl_size>
CtrlMatrix CostFunction<state_size, ctrl_size>::L_uu(const mjData *d)
{
    update_errors(d);
    return 2 * m_u_gain * 2;
}


template<int state_size, int ctrl_size>
CtrlStateMatrix CostFunction<state_size, ctrl_size>::L_ux(const mjData *d)
{
    update_errors(d);
    return Eigen::Matrix<mjtNum, ctrl_size, state_size>::Zero();
}

using namespace SimulationParameters;
template class CostFunction<state_size, n_ctrl>;