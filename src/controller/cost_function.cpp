#include <iostream>
#include "cost_function.h"
#include "../utilities/mujoco_utils.h"


using namespace InternalTypes;

CostFunction::CostFunction(const StateVector& x_desired,
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


void CostFunction::update_errors(const mjData *d)
{
    MujocoUtils::fill_state_vector(d, m_x, m_m);
    MujocoUtils::fill_ctrl_vector(d, m_u, m_m);
    m_x_error = m_x - m_x_desired;
    m_u_error = m_u;
    m_du_error = m_u - m_u_prev;
    m_u_prev = m_u;
}


inline void CostFunction::update_errors(const StateVector& state, const CtrlVector& ctrl)
{
//    for(unsigned int row = 0; row < state_size/2; ++row)
//    {
//        state(row, 0) = state(row, 0);
//    }
    m_x_error = state - m_x_desired;
    m_u_error = ctrl;
}


mjtNum CostFunction::running_cost(const mjData *d)
{
    update_errors(d);
    return (m_x_error.transpose() * m_x_gain * m_x_error)(0, 0) +
           (m_u_error.transpose() * m_u_gain * m_u_error)(0, 0) +
           (m_du_error.transpose() * m_u_diff_gain * m_du_error)(0, 0);
}


mjtNum CostFunction::trajectory_running_cost(const std::vector<StateVector>& x_trajectory,
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


mjtNum CostFunction::terminal_cost(const mjData *d)
{
    update_errors(d);
    return (m_x_error.transpose() * m_x_terminal_gain * m_x_error)(0, 0);
}


StateVector CostFunction::Lf_x(const mjData *d)
{
    update_errors(d);
    return m_x_error.transpose() * (2 * m_x_terminal_gain);
}


StateMatrix CostFunction::Lf_xx()
{
    return 2 * m_x_terminal_gain;
}


StateVector CostFunction::L_x(const mjData *d)
{
    update_errors(d);
    return m_x_error.transpose() * (2 * m_x_gain);
}


StateMatrix CostFunction::L_xx(const mjData *d)
{
    update_errors(d);
    return 2 * m_x_gain;
}


CtrlVector CostFunction::L_u(const mjData *d)
{
    update_errors(d);
    return (m_u_error.transpose() * (2 * m_u_gain)) * 2;
}


CtrlMatrix CostFunction::L_uu(const mjData *d)
{
    update_errors(d);
    return 2 * m_u_gain * 2;
}


CtrlStateMatrix CostFunction::L_ux(const mjData *d)
{
    update_errors(d);
    return CtrlStateMatrix::Zero();
}
