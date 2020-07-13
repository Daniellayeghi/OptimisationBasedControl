#include <iostream>
#include "MPPI.h"
#include"simulation_params.h"
#include "../utilities/basic_math.h"

using namespace SimulationParameters;

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


    template<int state_size>
    inline void fill_state_vector(mjData* data, Eigen::Matrix<double, state_size, 1>& state)
    {
        for(auto row = 0; row < state.rows()/2; ++row)
        {
            state(row, 0) = BasicMath::wrap_to_min_max(data->qpos[row],-M_PI, M_PI);;
            state(row+state.rows()/2, 0) = data->qvel[row];
        }
    }


    template<int ctrl_size>
    void set_control_data(mjData* data, const Eigen::Matrix<double, ctrl_size, 1>& ctrl)
    {
        for(auto row = 0; row < ctrl.rows(); ++row)
        {
            data->ctrl[row] = ctrl(row, 0);
        }
    }


    template<int ctrl_size>
    void clamp_control(Eigen::Matrix<mjtNum, ctrl_size, 1>& control, mjtNum max_bound, mjtNum min_bound)
    {
        for (auto row = 0; row < control.rows(); ++row)
        {
            control(row, 0) = std::clamp(control(row, 0), min_bound, max_bound);
        }
    }
}


template<int state_size, int ctrl_size>
MPPI<state_size, ctrl_size>::MPPI(const mjModel *m, const QRCost<state_size, ctrl_size>& cost_func, const MPPIParams& params)
:
m_params(params),
m_cost_func(cost_func),
m_m(m)
{
    m_d_cp = mj_makeData(m_m);

    _cached_control = MPPI<state_size, ctrl_size>::ctrl_vector::Zero();

    m_state.assign(m_params.m_sim_time, Eigen::Matrix<double, state_size, 1>::Zero());
    m_control.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    m_delta_control.assign(m_params.m_sim_time,
            std::vector<Eigen::Matrix<double, ctrl_size, 1>>(m_params.m_k_samples,Eigen::Matrix<double, ctrl_size, 1>::Zero()));

    m_delta_cost_to_go.assign(m_params.m_k_samples,0);
}


template<int state_size, int ctrl_size>
typename MPPI<state_size, ctrl_size>::ctrl_vector
MPPI<state_size, ctrl_size>::total_entropy(const std::vector<MPPI<state_size, ctrl_size>::ctrl_vector>& delta_control_samples,
                                           const std::vector<double>& d_cost_to_go_samples) const
{
    MPPI<state_size, ctrl_size>::ctrl_vector numerator = MPPI<state_size, ctrl_size>::ctrl_vector::Zero();
    double denomenator =  0;
    for (auto& sample: d_cost_to_go_samples)
    {
        denomenator += (std::exp(-(1 / m_params.m_lambda) * sample));
    }

    for (unsigned long col = 0; col < d_cost_to_go_samples.size(); ++col)
    {
        numerator += (std::exp(-(1 / m_params.m_lambda) * d_cost_to_go_samples[col]) * delta_control_samples[col]);
    }
    return numerator/denomenator;
}


template <int state_size, int ctrl_size>
void MPPI<state_size, ctrl_size>::MPPI::compute_control_trajectory()
{
    for (auto time = 0; time < m_params.m_sim_time; ++time)
    {
        m_control[time] += (total_entropy(m_delta_control[time], m_delta_cost_to_go));
    }

    _cached_control = m_control.front();

    std::rotate(m_control.begin(), m_control.begin() + 1, m_control.end());
    m_control.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
}


template<int state_size, int ctrl_size>
void MPPI<state_size, ctrl_size>::control(const mjData* d)
{
    MPPI<state_size, ctrl_size>::ctrl_vector instant_control;

    fill_state_vector(m_d_cp, m_state.front());
    std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);

    for(auto sample = 0; sample < m_params.m_k_samples; ++sample)
    {
        copy_data(m_m, d, m_d_cp);
        for (auto time = 0; time < m_params.m_sim_time - 1; ++time)
        {
            // u += du -> du ~ N(0, variance)
            m_delta_control[time][sample] = m_params.m_variance * MPPI<state_size, ctrl_size>::ctrl_vector::Random();
            instant_control = m_control[time] + m_delta_control[time][sample];

            // Forward simulate controls
            set_control_data(m_d_cp, instant_control);
            mj_step(m_m, m_d_cp);
            fill_state_vector(m_d_cp, m_state[time + 1]);

            // Compute cost-to-go of the controls
            m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] + m_cost_func(m_state[time + 1],
                                                                                  m_control[time],
                                                                                  m_delta_control[time][sample],
                                                                                   m_params.m_variance);
        }
        m_delta_control.back()[sample] = m_params.m_variance * MPPI<state_size, ctrl_size>::ctrl_vector::Random();
    }

    compute_control_trajectory();
}


template<int state_size, int ctrl_size>
MPPI<state_size, ctrl_size>::~MPPI()
{
    mj_deleteData(m_d_cp);
}

template class MPPI<n_jpos + n_jvel, n_ctrl>;
