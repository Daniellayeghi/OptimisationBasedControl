#include <iostream>
#include "MPPI.h"
#include "Eigen/Core"

namespace
{
    template <typename T>
    void copy_data(const mjModel * model, const mjData *data, T *data_cp)
    {
        data_cp->time = data->time;
        mju_copy(data_cp->qpos, data->qpos, model->nq);
        mju_copy(data_cp->qvel, data->qvel, model->nv);
        mju_copy(data_cp->qacc, data->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data->xfrc_applied, 6*model->nbody);
        mju_copy(data_cp->ctrl, data->ctrl, model->nu);
    }


    template<typename T, int M, int N>
    void fill_state_vector(mjData* data, Eigen::Matrix<T, M, N>& state)
    {
        state(0, 0) = data->qpos[0]; state(1, 0) = data->qpos[1];
        state(2, 0) = data->qvel[0]; state(3, 0) = data->qvel[1];
    }


    template<typename T, int M, int N>
    void set_control_data(mjData* data, const Eigen::Matrix<T, M, N>& ctrl)
    {
        for(auto row = 0; row < ctrl.rows(); ++row)
        {
            data->ctrl[row] = ctrl(row, 0);
        }
    }


    template<int rows, int cols>
    void clamp_control(Eigen::Matrix<mjtNum, rows, cols>& control, mjtNum max_bound, mjtNum min_bound)
    {
        for (auto row = 0; row < control.rows(); ++row)
        {
            control(row, 0) = std::clamp(control(row, 0), min_bound, max_bound);
        }
    }
}


MPPI::MPPI(const mjModel *m) : _m(m)
{
    _d_cp = mj_makeData(_m);

    _Q_state_cost << 6, 0, 0 ,0,
                     0, 6, 0, 0,
                     0, 0, 0.01, 0,
                     0, 0, 0, 0.01;

    _R_control_cost << 1, 0,
                       0, 1;
    _R_control_cost *= 10;

    _cached_control << 0, 0;

    _state.assign(_sim_time,Eigen::Matrix<double, 4, 1>::Zero());
    _control.assign(_sim_time, Eigen::Matrix<double, 2, 1>::Zero());
    _delta_cost_to_go.assign(_k_samples, 0);

    for (auto time = 0; time < _sim_time; ++time)
    {
        _delta_control[time].assign(_k_samples, Eigen::Matrix<double, 2, 1>::Zero());
    }
}


double MPPI::q_cost(Mat4x1& state)
{
    fill_state_vector(_d_cp, state);
    return state.transpose() * _Q_state_cost * state;
}


double MPPI::delta_q_cost(Mat4x1& state, Mat2x1& du, Mat2x1& u)
{
    fill_state_vector(_d_cp, state);
    return q_cost(state) + du.transpose() * _R_control_cost * du + u.transpose() * _R_control_cost * du +
           0.5 * u.transpose() * _R_control_cost *  u;
}


Mat2x1 MPPI::total_entropy(const std::vector<Mat2x1>& delta_control_samples)
{
    Mat2x1 numerator; numerator << 0, 0;
    double denomenator =  0;
    for (unsigned long col = 0; col < _delta_cost_to_go.size(); ++col)
    {
        numerator += (std::exp(-(1/_lambda) * _delta_cost_to_go[col]) * delta_control_samples[col]);
        denomenator += (std::exp(-(1/_lambda) * _delta_cost_to_go[col]));
    }

    return numerator/denomenator;
}


void MPPI::control(const mjData* d)
{
    using namespace InternalTypes;
    Eigen::Matrix<double, 2, 1> instant_control;
    fill_state_vector(_d_cp, _state[0]);
    _delta_cost_to_go.assign(_k_samples, 0);
    for(auto sample = 0; sample < _k_samples; ++sample)
    {
        copy_data(_m, d, _d_cp);
        for (auto time = 0; time < _sim_time - 1; ++time)
        {
            _delta_control[time][sample] = _variance * Mat2x1::Random();
            instant_control = _control[time] + _delta_control[time][sample];
            set_control_data(_d_cp, instant_control);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _state[time+1]);
            _delta_cost_to_go[sample] += delta_q_cost(_state[time+1], _delta_control[time][sample], _control[time]);
        }
        _delta_control.back()[sample] = _variance * Mat2x1::Random();
    }

    for (auto time = 0; time < _sim_time; ++time)
    {
        _control[time] += (total_entropy(_delta_control[time]));
        clamp_control(_control[time], 1, -1);
    }

    _cached_control = _control[0];

    for (auto time = 0; time < _sim_time - 1; ++time)
    {
        _control[time] = _control[time + 1];
    }
    _control.back() = Mat2x1::Ones();
}

