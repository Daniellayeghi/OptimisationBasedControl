#include <iostream>
#include "cost_function.h"
#include "../utilities/mujoco_utils.h"


using namespace InternalTypes;

template<int state_size, int ctrl_size>
CostFunction<state_size, ctrl_size>::CostFunction(const StateVector& x_desired,
                                                  const CtrlVector& u_desired,
                                                  const StateMatrix& x_gain,
                                                  const CtrlMatrix& u_gain,
                                                  const StateMatrix& x_terminal_gain,
                                                  const mjModel* m)
                                                  :
                                                  _u_gain(u_gain),
                                                  _x_gain(x_gain),
                                                  _x_terminal_gain(x_terminal_gain),
                                                  _u_desired(u_desired),
                                                  _x_desired(x_desired),
                                                  _m(m)
{

}


template<int state_size, int ctrl_size>
void CostFunction<state_size, ctrl_size>::update_errors(const mjData *d)
{
    MujocoUtils::fill_state_vector(d, _x);
    MujocoUtils::fill_ctrl_vector(d, _u);
    _x_error = _x - _x_desired;
    _u_error = _u;
}


template<int state_size, int ctrl_size>
inline void CostFunction<state_size, ctrl_size>::update_errors(StateVector& state,
                                                               CtrlVector& ctrl)
{
    for(unsigned int row = 0; row < state_size/2; ++row)
    {
        state(row, 0) = state(row, 0);
    }
    _x_error = state - _x_desired;
    _u_error = ctrl;
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::running_cost(const mjData *d)
{
    update_errors(d);
    return (_x_error.transpose() * _x_gain * _x_error)(0,0) +
           (_u_error.transpose() * _u_gain * _u_error)(0, 0);
}


template<int state_size, int ctrl_size>
inline mjtNum CostFunction<state_size, ctrl_size>::trajectory_running_cost(std::vector<StateVector>& x_trajectory,
                                                                           std::vector<CtrlVector>& u_trajectory)
{
    auto cost = 0.0;
    //Compute running cost
    for(unsigned int row = 0; row < u_trajectory.size(); ++row)
    {
        update_errors(x_trajectory[row], u_trajectory[row]);
        cost += (_x_error.transpose() * _x_gain * _x_error)(0,0) +
                (_u_error.transpose() * _u_gain * _u_error)(0,0);
    }

    //Compute terminal cost
    for(unsigned int row = 0; row < state_size/2; ++row)
    {
        int jid = _m->dof_jntid[row];
        if(_m->jnt_type[jid] == mjJNT_HINGE)
            x_trajectory.back()(row, 0) = x_trajectory.back()(row, 0);
    }
    _x_error =  _x_desired - x_trajectory.back();
    //Running cost + terminal cost
    return cost + (_x_error.transpose() * _x_terminal_gain * _x_error)(0, 0);
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::terminal_cost(const mjData *d)
{
    update_errors(d);
    return (_x_error.transpose() * _x_terminal_gain * _x_error)(0, 0);
}


template<int state_size, int ctrl_size>
StateVector CostFunction<state_size, ctrl_size>::Lf_x(const mjData *d)
{
    update_errors(d);
    return  _x_error.transpose() * (2 *_x_terminal_gain);
}


template<int state_size, int ctrl_size>
StateMatrix CostFunction<state_size, ctrl_size>::Lf_xx()
{
    return  2 *_x_terminal_gain;
}


template<int state_size, int ctrl_size>
StateVector CostFunction<state_size, ctrl_size>::L_x(const mjData *d)
{
    update_errors(d);
    return _x_error.transpose() * (2 * _x_gain);
}


template<int state_size, int ctrl_size>
StateMatrix CostFunction<state_size, ctrl_size>::L_xx(const mjData *d)
{
    update_errors(d);
    return 2 * _x_gain;
}


template<int state_size, int ctrl_size>
CtrlVector CostFunction<state_size, ctrl_size>::L_u(const mjData *d)
{
    update_errors(d);
    return (_u_error.transpose() * (2 * _u_gain));
}


template<int state_size, int ctrl_size>
CtrlMatrix CostFunction<state_size, ctrl_size>::L_uu(const mjData *d)
{
    update_errors(d);
    return 2 * _u_gain;
}


template<int state_size, int ctrl_size>
CtrlStateMatrix CostFunction<state_size, ctrl_size>::L_ux(const mjData *d)
{
    update_errors(d);
    return Eigen::Matrix<mjtNum, ctrl_size, state_size>::Zero();
}

using namespace SimulationParameters;
template class CostFunction<state_size, n_ctrl>;