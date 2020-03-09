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


    void fill_data(Eigen::Matrix<double, 4, 1>& _u, Eigen::Matrix<double, 4, 1>& _x, mjData* _state)
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
    inline T running(Eigen::Matrix<T, 4, 1> &x, Eigen::Matrix<T, 4, 1> &u)
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


CostFunction::CostFunction()
{
    _u.setZero();
    _x.setZero();
    _Ax.setZero();
    _Au.setZero();
    _gradient.setZero();
    _hessian.setZero();
}


void CostFunction::derivatives(mjData* d)
{
    using namespace AutoDiffTypes;

    fill_data(_u, _x, d);
    // copy value from non-active example
    for(int row = 0; row < _x.size(); ++row)
        _Ax(row).value().value() = _x(row);

    for(int row = 0; row < _u.size(); ++row)
        _Au(row).value().value() = _u(row);

    // initialize derivative vectors
    auto derivative_num = _x.size() + _u.size();
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
////#if DEFINE_DEBUG
//    std::cout << "Hessian:" << "\n" << _hessian.block<6, 6>(0, 0) << "\n";
////#endif
}



Eigen::Ref<Block<Eigen::Matrix<double, 8, 1>, 4, 1>> CostFunction::L_x()
{
    return _gradient.block<4, 1>(0, 0);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 1>, 2, 1>> CostFunction::L_u()
{
    return _gradient.block<2, 1>(4, 0);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 6, 6>>  CostFunction::L_xx()
{
    return _hessian.block<6, 6>(0, 0);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 2, 2>> CostFunction::L_uu()
{
    return _hessian.block<2, 2>(4, 4);
}


Eigen::Ref<Block<Eigen::Matrix<double, 8, 8, 0, 8, 8>, 2, 4>> CostFunction::L_ux()
{
    return _hessian.block<2, 4>(4, 0);
}
