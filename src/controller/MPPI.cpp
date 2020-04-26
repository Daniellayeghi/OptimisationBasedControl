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

    _Q_state_cost << 12, 0, 0 ,0,
                     0, 12, 0, 0,
                     0, 0, 0.1, 0,
                     0, 0, 0, 0.1;

    _R_control_cost << 500, 0,
                       0, 500;
    _R_control_cost *= 1;

    _cached_control << 0, 0;

    _state.assign(_sim_time,Eigen::Matrix<double, 4, 1>::Zero());
    _control.assign(_sim_time, Eigen::Matrix<double, 2, 1>::Zero());

    for (auto time = 0; time < _sim_time; ++time)
    {
        _delta_control[time].assign(_k_samples, Eigen::Matrix<double, 2, 1>::Zero());
        _delta_cost_to_go[time].assign(_k_samples, 0);
    }
}


double MPPI::q_cost(Mat4x1 state)
{;
    state(0,0) =  1 - sin(state(0, 0));
    state(1,0) =  1 - cos(state(1, 0));
    std::cout << "state_0: " << 1 - sin(state(0, 0)) << std::endl;
    std::cout << "state_1: " << 1 - cos(state(1, 0)) << std::endl;
    return state.transpose() * _Q_state_cost * state;
}


double MPPI::delta_q_cost(Mat4x1& state, Mat2x1& du, Mat2x1& u)
{
    u(0,0) = 0;
    std::cout << "cost: " << q_cost(state) << std::endl;
    return q_cost(state);
}


Mat2x1 MPPI::total_entropy(const std::vector<Mat2x1>& delta_control_samples,
                           const std::vector<double>& d_cost_to_go_samples) const
{
    Mat2x1 numerator; numerator << 0, 0;
    double denomenator =  0;
    for (auto& sample: d_cost_to_go_samples)
    {
        denomenator += (std::exp(-(1/_lambda) * sample));
    }

    for (unsigned long col = 0; col < d_cost_to_go_samples.size(); ++col)
    {
        numerator += (std::exp(-(1/_lambda) * d_cost_to_go_samples[col]) * delta_control_samples[col]);
    }
    Mat2x1 result = numerator/denomenator; result(0,0) = 0;
    return result;
}


void MPPI::control(const mjData* d)
{
    using namespace InternalTypes;
    Eigen::Matrix<double, 2, 1> instant_control;
    fill_state_vector(_d_cp, _state[0]);
    _delta_cost_to_go.front().assign(_k_samples, 0);
    for(auto sample = 0; sample < _k_samples; ++sample)
    {
        copy_data(_m, d, _d_cp);
        for (auto time = 0; time < _sim_time - 1; ++time)
        {
            _delta_control[time][sample] = _variance * Mat2x1::Random();
            _delta_control[time][sample](0, 0) = 0;
            instant_control = _control[time] + _delta_control[time][sample];
            clamp_control(instant_control, 2, -2);
            set_control_data(_d_cp, instant_control);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _state[time+1]);
            _delta_cost_to_go[time+1][sample] = _delta_cost_to_go[time][sample] +
                    (delta_q_cost(_state[time+1], _delta_control[time][sample], _control[time]));
        }
        _delta_control.back()[sample] = _variance * Mat2x1::Random();
    }

    for (auto time = 0; time < _sim_time - 1; ++time)
    {
        _control[time] += (total_entropy(_delta_control[time], _delta_cost_to_go[time]));
        clamp_control(_control[time], 2, -2);
    }

    _cached_control = _control[0];

    for (auto time = 0; time < _sim_time - 1; ++time)
    {
        _control[time] = _control[time + 1];
    }
    _control.back() = Mat2x1::Ones();
}

