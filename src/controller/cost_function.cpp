#include <iostream>
#include "cost_function.h"
#include "simulation_params.h"
#include "../utilities/basic_math.h"

namespace
{

#if 0
    template <typename T>
    inline void init_twice_active_var(T &ad,int d_num, int idx)
    {
        // initialize derivative direction in value field of outer active variable
        ad.value().derivatives() = T::DerType::Scalar::DerType::Unit(d_num, idx);
        // initialize derivatives direction of the variable
        ad.derivatives() = T::DerType::Unit(d_num, idx);
        // initialize Hessian matrix of variable to zero

        for (int index = 0; index < d_num; index++)
        {
            ad.derivatives()(index).derivatives() = T::DerType::Scalar::DerType::Zero(d_num);
        }
    }
#endif

    template<int state_size, int ctrl_size>
    void fill_data(Eigen::Matrix<mjtNum , state_size, 1>& u, Eigen::Matrix<mjtNum, ctrl_size, 1>&x, const mjData* state)
    {

        for(unsigned int row = 0; row < state_size/2; ++row)
        {
            x(row, 0) = BasicMath::wrap_to_min_max(state->qpos[row],-M_PI, M_PI);
            x(row + state_size/2, 0) = state->qvel[row];
        }

        for(unsigned int row = 0; row < ctrl_size; ++row)
        {
            u(row, 0) = state->ctrl[row];
        }
    }
}


using namespace InternalTypes;

template<int state_size, int ctrl_size>
CostFunction<state_size, ctrl_size>::CostFunction(const mjData* d,
                           const state_vec& x_desired,
                           const ctrl_vec& u_desired,
                           const state_mat& x_gain,
                           const ctrl_mat& u_gain,
                           const state_mat& x_terminal_gain) : _d(d)
{
    _u_desired = u_desired;
    _x_desired = x_desired;
    _x_gain = x_gain;
    _u_gain = u_gain;
    _x_terminal_gain = x_terminal_gain;
}


template<int state_size, int ctrl_size>
void CostFunction<state_size, ctrl_size>::update_errors()
{
    fill_data(_u, _x, _d);
    _x_error = _x_desired - _x;
    _u_error = _u;
}


template<int state_size, int ctrl_size>
inline void CostFunction<state_size, ctrl_size>::update_errors(state_vec& state,
                                                               ctrl_vec& ctrl)
{
    for(unsigned int row = 0; row < state_size/2; ++row)
    {
        state(row, 0) = BasicMath::wrap_to_min_max(state(row, 0),-M_PI, M_PI);
    }
    _x_error = _x_desired - state;
    _u_error = ctrl;
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::running_cost()
{
    update_errors();
    return (_x_error.transpose() * _x_gain * _x_error)(0,0) +
           (_u_error.transpose() * _u_gain * _u_error)(0, 0);
}


template<int state_size, int ctrl_size>
inline mjtNum CostFunction<state_size, ctrl_size>::trajectory_running_cost(std::vector<state_vec>& x_trajectory,
                                                                           std::vector<ctrl_vec>& u_trajectory)
{
    auto cost = 0.0;
    for(unsigned int row = 0; row < u_trajectory.size(); ++row)
    {
        update_errors(x_trajectory[row], u_trajectory[row]);
        cost += (_x_error.transpose() * _x_gain * _x_error)(0,0) +
                (_u_error.transpose() * _u_gain * _u_error)(0,0);
    }
    return cost;
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::terminal_cost()
{
    update_errors();
    return (_x_error.transpose() * _x_terminal_gain * _x_error)(0, 0);
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, 1> CostFunction<state_size, ctrl_size>::Lf_x()
{
    update_errors();
    return  _x_error.transpose() * (2 *_x_terminal_gain);
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, state_size> CostFunction<state_size, ctrl_size>::Lf_xx()
{
    return  2 *_x_terminal_gain;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, 1> CostFunction<state_size, ctrl_size>::L_x()
{
    update_errors();
    return _x_error.transpose() * (2 * _x_gain);
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, state_size> CostFunction<state_size, ctrl_size>::L_xx()
{
    update_errors();
    return 2 * _x_gain;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, ctrl_size, 1> CostFunction<state_size, ctrl_size>::L_u()
{
    update_errors();
    return (_u_error.transpose() * (2 * _u_gain)).transpose();
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, ctrl_size, ctrl_size> CostFunction<state_size, ctrl_size>::L_uu()
{
    update_errors();
    return 2 * _u_gain;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, ctrl_size, state_size> CostFunction<state_size, ctrl_size>::L_ux()
{
    update_errors();
    return Eigen::Matrix<mjtNum, ctrl_size, state_size>::Zero();
}

#if 0
Eigen::Ref<Block<Eigen::Matrix<double, 8, 1>, 2, 1>> CostFunction::L_u()
{
    return _gradient.block<2, 1>(4, 0);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 4, 4>>  CostFunction::L_xx()
{
    return _hessian.block<4, 4>(0, 0);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 2, 2>> CostFunction::L_uu()
{
    return _hessian.block<2, 2>(4, 4);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 2, 4>> CostFunction::L_ux()
{
    return _hessian.block<2, 4>(4, 0);
}


void CostFunction::derivatives(const mjData* d)
{
    using namespace AutoDiffTypes;

    fill_data(_u_desired, _x_desired, d);
    // copy value from non-active example
    for(int row = 0; row < _x_desired.size(); ++row)
        _Ax(row).value().value() = _x_desired(row);

    for(int row = 0; row < _u_desired.size(); ++row)
        _Au(row).value().value() = _u_desired(row);

    // initialize derivative vectors
    auto derivative_num = _x_desired.size() + _u_desired.size();
    int derivative_idx = 0;

    for(int row = 0; row < _Ax.size(); ++row)
    {
        init_twice_active_var(_Ax(row), derivative_num, derivative_idx);
        derivative_idx++;
    }

    for(int row=0; row < _Au.size(); row++)
    {
        init_twice_active_var(_Au(row, 0), derivative_num, derivative_idx);
        derivative_idx++;
    }

    _Ac = running(_Ax, _Au);
    _gradient = _Ac.value().derivatives().transpose();

    for(int idx = 0; idx < _Ac.derivatives().size(); ++idx)
    {
        _hessian.middleRows(idx,1) = _Ac.derivatives()(idx).derivatives().transpose();
    }

#if DEFINE_DEBUG
    std::cout << "Hessian:" << "\n" << _hessian.block<6, 6>(0, 0) << "\n";
#endif
}
#endif

using namespace SimulationParameters;
template class CostFunction<n_jpos + n_jvel, n_ctrl>;