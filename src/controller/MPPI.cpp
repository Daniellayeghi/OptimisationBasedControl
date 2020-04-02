#include "MPPI.h"
#include "Eigen/Core"

namespace
{
    template <typename T>
    void copy_data(const mjModel* model, const mjData *data, T *data_cp)
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
    void fill_state_vector(mjData* data, Eigen::Matrix<T, M, N> state)
    {
        state(0, 0) = data->qpos[0]; state(1, 0) = data->qpos[1];
        state(2, 0) = data->qvel[0]; state(3, 0) = data->qvel[1];
    }


    template<typename T, int M, int N>
    void set_control_data(mjData* data, const Eigen::Matrix<T, M, N>& ctrl)
    {
        for(auto row = 0; row < ctrl.rows(); ++row)
        {
            data->qfrc_applied[row] = ctrl(row, 0);
        }
    }
}


MPPI::MPPI(mjModel *m) : _m(m)
{
    _Q_state_cost << 1, 0, 0 ,0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1;

    _R_control_cost << 1, 0,
                       0, 1;

    _state.assign(500,Eigen::Matrix<double, 4, 1>::Zero());
    _control.assign(500, Eigen::Matrix<double, 2, 1>::Zero());

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


void MPPI::control(const mjData* d)
{
    using namespace InternalTypes;
    Mat4x1 state;

    copy_data(_m, d, _d_cp);
    fill_state_vector(_d_cp, _state[0]);
    for(auto sample = 0; sample < _k_samples; ++sample)
    {
        for (auto time = 0; time < _sim_time - 1; ++time)
        {
            _delta_control(time, sample) = _variance * Mat2x1::Random();
            _control[time] += _delta_control(time, sample);
            set_control_data(_d_cp, _control[time]);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _state[time]);
        }
    }
}

