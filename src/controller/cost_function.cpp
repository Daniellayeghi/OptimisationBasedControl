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


    void fill_data(Eigen::Matrix<double, 2, 1>& _u,
                   Eigen::Matrix<double, 4, 1>& _x,
                   mjData* _state)
    {
        _x(0, 0) = _state->qpos[0];
        _x(1, 0) = _state->qpos[1];
        _x(2, 0) = _state->qvel[0];
        _x(3, 0) = _state->qvel[1];
        _u(0, 0) = _state->ctrl[0];
        _u(1, 0) = _state->ctrl[1];
    }



    template <typename T>
    inline T running(Eigen::Matrix<T, 4, 1> &x, Eigen::Matrix<T, 4, 1> &u)
    {
        Eigen::Matrix<T, 2, 1> gain_state; gain_state.setOnes();
        Eigen::Matrix<T, 2, 1> gain_action; gain_action.setOnes();

        return (x.transpose() * x + u.transpose() * u)(0, 0);
    }


    template <typename T>
    T terminal (Eigen::Matrix<T, 4, 1> &x, Eigen::Matrix<T, 2, 1> &u)
    {
        return 1000.0;
    }


    template <typename T>
    T my_matrixfun(Eigen::Matrix<T, 4, 1> const &a)
    {
        Eigen::Matrix<T, 4, 1> gain;
        for (auto row = 0; row < gain.rows(); ++row){gain(row) = 2;};
        return ( a.transpose() * gain.asDiagonal() * a)(0,0);
    }


}


CostFunction::CostFunction(mjData *state)
{
    _state = state;
//    _u.resize(4, 1);
//    _x.resize(4, 1);
//    _Ax.resize(_x.size());
//    _Au.resize(_u.size());
//    _gradient.resize(6, 1);
//    _hessian.resize(6, 6);
}


VectorXd CostFunction::Lf_x()
{
    using namespace AutoDiffTypes;
    fill_data(_u, _x, _state);

    // copy value from non-active example
    for(int row = 0; row < _x.size(); ++row)
        _Ax(row).value().value() = _x(row);

    for(int row = 0; row < _u.size(); ++row)
        _Au(row).value().value() = _u(row);

//  initialize derivative vectors
    auto derivative_num = _x.size() + _u.size() - 2;
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
//
//    for(int idx = 0; idx < _Ac.derivatives().size(); ++idx)
//    {
//        _hessian.middleRows(idx,1) = _Ac.derivatives()(idx).derivatives().transpose();
//    }
#ifdef MUJ_KEY_PATH
    std::cout << "Gradient:" << _Ac.value().derivatives().transpose() << "\n";
    std::cout << "Hessian:" << "\n" << _hessian << "\n";
#endif
}


//
//Vector4d CostFunction::L_x()
//{
//    dual running_cost;
//    return gradient(_running_cost, wrt(_x), at(_x, _u), running_cost);
//}


/*Vector2d CostFunction::L_u()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_u), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_xx()
{
    dual running_cost;
    return gradient(_running_cost, wrt<2>(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::L_ux()
{
    dual running_cost;
    return gradient(_running_cost, wrt(_u, _x), at(_x, _u), running_cost);
}


dual CostFunction::Lf()
{
    return _terminal_cost(_x, _u);
}


VectorXd CostFunction::Lf_x()
{
    dual running_cost;
    return gradient(_terminal_cost, wrt(_x), at(_x, _u), running_cost);
}


VectorXd CostFunction::Lf_xx()
{
    dual running_cost;
    return gradient(_terminal_cost, wrt<2>(_x), at(_x, _u), running_cost);
}*/
