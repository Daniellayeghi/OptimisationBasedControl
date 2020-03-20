#include <iostream>
#include "cost_function.h"
#include "../utilities/internal_types.h"

namespace
{
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


    void fill_data(Eigen::Matrix<double, 2, 1>& _u, Eigen::Matrix<double, 4, 1>& _x, const mjData* _state)
    {
        _x(0, 0) = _state->qpos[0];
        _x(1, 0) = _state->qpos[1];
        _x(2, 0) = _state->qvel[0];
        _x(3, 0) = _state->qvel[1];
        _u(0, 0) = _state->ctrl[0];
        _u(1, 0) = _state->ctrl[1];
        _u(2, 0) = 0;
        _u(3, 0) = 0;
    }


    template <typename T>
    inline T running(Eigen::Matrix<T, 4, 1> &x, Eigen::Matrix<T, 2, 1> &u)
    {
        Eigen::Matrix<T, 4, 1> gain_state; gain_state.setOnes();
        Eigen::Matrix<T, 4, 1> gain_action; gain_action.setOnes();
        for (auto row = 0; row < gain_action.rows(); ++row){gain_action(row) *= 2;}

        return (x.transpose() * gain_state.asDiagonal() * x + u.transpose() * gain_action.asDiagonal() * u)(0, 0);
    }


    template <typename T>
    T terminal (Eigen::Matrix<T, 4, 1> &x, Eigen::Matrix<T, 4, 1> &u)
    {
        return 1000.0;
    }
}


using namespace InternalTypes;

CostFunction::CostFunction(const mjData* d,
                           const Mat4x1& x_desired,
                           const Mat2x1& u_desired,
                           const Mat4x4& x_gain,
                           const Mat2x2& u_gain,
                           const Mat4x4& x_terminal_gain) : _d(d)
{
    _u_desired = u_desired;
    _x_desired = x_desired;

    _x_gain = x_gain;
    _u_gain = u_gain;
    _x_terminal_gain = x_terminal_gain;
}


void CostFunction::update_errors()
{
    _x_error(0, 0) = _x_desired(0, 0) - _d->qpos[0];
    _x_error(1, 0) = _x_desired(1, 0) - _d->qpos[1];
    _x_error(2, 0) = _x_desired(2, 0) - _d->qvel[0];
    _x_error(3, 0) = _x_desired(3, 0) - _d->qvel[1];
    _u_error(0, 0) = _u_desired(0, 0) - _d->ctrl[0];
    _u_error(1, 0) = _u_desired(1, 0) - _d->ctrl[1];
}


template<int x_rows, int u_rows, int cols>
inline void CostFunction::update_errors(const Eigen::Matrix<mjtNum, x_rows, cols> &state,
                                        const Eigen::Matrix<mjtNum, u_rows, cols> &ctrl)
{
    _x_error = _x_desired - state;
    _u_error = _u_desired - ctrl;
}


mjtNum CostFunction::running_cost()
{
    update_errors();
    return (_x_error.transpose() * _x_gain * _x_error + _u_error.transpose() * _u_gain * _u_error)(0, 0);
}


template<int x_rows, int u_rows, int cols>
inline mjtNum CostFunction::trajectory_running_cost(const std::vector<Eigen::Matrix<mjtNum, x_rows, cols>> & x_trajectory,
                                                    const std::vector<Eigen::Matrix<mjtNum, u_rows, cols>> & u_trajectory)
{
    auto cost = 0.0;
    for(auto row = 0; row < u_trajectory.size(); ++row)
    {
        update_errors(x_trajectory[row], u_trajectory[row]);
        cost += (_x_error.transpose() * _x_gain * _x_error + _u_error.transpose() * _u_gain * _u_error)(0, 0);
    }
    return cost;
}


mjtNum CostFunction::terminal_cost()
{
    update_errors();
    return (_x_error.transpose() * _x_terminal_gain * _x_error)(0, 0);
}


Eigen::Matrix<mjtNum, 4, 1> CostFunction::Lf_x()
{
    update_errors();
    return  _x_error.transpose() * (2 *_x_terminal_gain);
}


Eigen::Matrix<mjtNum, 4, 4> CostFunction::Lf_xx()
{
    update_errors();
    return  2 *_x_terminal_gain;
}


Eigen::Matrix<mjtNum, 4, 1> CostFunction::L_x()
{
    update_errors();
    return _x_error.transpose() * (2 * _x_gain);
}


Eigen::Matrix<mjtNum, 4, 4>  CostFunction::L_xx()
{
    update_errors();
    return 2 * _x_gain;
}


Eigen::Matrix<mjtNum, 2, 1> CostFunction::L_u()
{
    update_errors();
    return _u_error.transpose() * (2 * _u_gain);
}


Eigen::Matrix<mjtNum, 2, 2> CostFunction::L_uu()
{
    update_errors();
    return 2 * _u_gain;
}


Eigen::Matrix<mjtNum, 2, 4> CostFunction::L_ux()
{
    update_errors();
    return Eigen::Matrix<mjtNum, 2, 4>::Zero();
}


template mjtNum CostFunction::trajectory_running_cost<4, 2, 1>(const std::vector<Eigen::Matrix<mjtNum, 4, 1> > &x_trajectory,
                                                               const std::vector<Eigen::Matrix<mjtNum, 2, 1> > &u_trajectory);

template void CostFunction::update_errors<4, 2, 1>(const Eigen::Matrix<mjtNum, 4, 1> &state,
                                                   const Eigen::Matrix<mjtNum, 2, 1> &ctrl);

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
