
#include "mppi_ddp.h"
#include "../../src/utilities/buffer.h"
#include "../../src/utilities/mujoco_utils.h"
#include <iostream>
#include <numeric>

using namespace SimulationParameters;
using namespace MujocoUtils;

template<int state_size, int ctrl_size>
MPPIDDP<state_size, ctrl_size>::MPPIDDP(const mjModel* m,
                                        QRCostDDP<state_size, ctrl_size>& cost,
                                        MPPIDDPParams<ctrl_size>& params):
        m_params(params),
        m_cost_func(cost),
        m_m(m),
        m_normX_cholesk(m_params.pi_ctrl_mean, params.ctrl_variance, m_params.m_sim_time, true)

{
    m_d_cp = mj_makeData(m_m);
    _cached_control = ctrl_vector::Zero();

    m_state.assign(m_params.m_sim_time, Eigen::Matrix<double, state_size, 1>::Zero());
    m_control.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    m_control_cp.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    m_delta_control.assign(m_params.m_sim_time,std::vector<Eigen::Matrix<double, ctrl_size, 1>>(
            m_params.m_k_samples,Eigen::Matrix<double, ctrl_size, 1>::Random()
    ));

    m_delta_cost_to_go.assign(m_params.m_k_samples,0);
    m_ctrl_samp_time.resize(m_params.m_k_samples, m_params.m_sim_time);
    m_ctrl_samples_time.resize(m_params.m_k_samples, m_params.m_sim_time * n_ctrl);
    m_cost_to_go_sample_time.assign(m_params.m_k_samples, std::vector<double>(m_params.m_sim_time, 0));
}


template<int state_size, int ctrl_size>
std::pair<typename MPPIDDP<state_size, ctrl_size>::ctrl_vector, typename MPPIDDP<state_size, ctrl_size>::ctrl_matrix>
MPPIDDP<state_size, ctrl_size>::total_entropy(const int time, const double min_cost) const
{

    // Computing the new covariance is taken from:
    // "Path Integral Policy Improvement with Covariance Matrix Adaptation"
    ctrl_vector numerator_mean = ctrl_vector::Zero();
    ctrl_matrix numerator_cov  = ctrl_matrix::Zero();
    double denomenator =  0;

    for (auto& sample_cost: m_delta_cost_to_go)
    {
        auto cost_diff = sample_cost - min_cost;
        denomenator += (std::exp(-(1 / m_params.m_lambda) * (cost_diff)));
    }

    auto ctrl_time_samples = m_ctrl_samples_time.transpose();
    auto numerator_weight = 0.0;

    for (unsigned long col = 0; col < m_delta_cost_to_go.size(); ++col)
    {
        numerator_weight = std::exp(-(1 / m_params.m_lambda) * (m_delta_cost_to_go[col] - min_cost));
        auto ctrl_sample = ctrl_time_samples.block(time*ctrl_size, col, ctrl_size, 1);
        numerator_mean += (numerator_weight * ctrl_sample);
        numerator_cov += (
                numerator_weight * (ctrl_sample - m_params.pi_ctrl_mean) *
                (ctrl_sample - m_params.pi_ctrl_mean).transpose()
                );
    }

    return {numerator_mean / (
            denomenator * std::pow(denomenator, - m_params.importance/(1+ m_params.importance)* m_params.m_scale)
            ), numerator_cov/denomenator};
}


template <int state_size, int ctrl_size>
void MPPIDDP<state_size, ctrl_size>::MPPIDDP::compute_control_trajectory()
{
    const auto min_cost = std::min_element(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end());
    double temp_mean_denomenator = 0;
    ctrl_vector new_mean_numerator = ctrl_vector::Zero(); ctrl_matrix new_cov_numerator = ctrl_matrix::Zero();
    Eigen::Matrix<double, n_ctrl, n_ctrl> new_variance; new_variance.setZero();

    for (auto time = 0; time < m_params.m_sim_time; ++time)
    {
        const auto [ctrl_pert, ctrl_variance] = (total_entropy(time, *min_cost));
        m_control[time] += ctrl_pert;

        // Temporal average for the mean and covariance
//        clamp_control(m_control[time], m_m->actuator_ctrlrange);
        ctrl_vector test_1; test_1.setRandom();  ctrl_vector test_2; test_2.setRandom();
        new_mean_numerator += (ctrl_pert * (m_params.m_sim_time - time));
        temp_mean_denomenator += (m_params.m_sim_time - time);
        new_variance += (ctrl_variance * (m_params.m_sim_time - time));
    }

    _cached_control = m_control.front();
    std::copy(m_control.begin(), m_control.end(), m_control_cp.begin());
    std::rotate(m_control.begin(), m_control.begin() + 1, m_control.end());
    m_control.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();

    // Do the averaging
    m_params.pi_ctrl_mean = new_mean_numerator / temp_mean_denomenator;
    m_params.ctrl_variance = new_variance / temp_mean_denomenator;
}


template<int state_size, int ctrl_size>
void MPPIDDP<state_size, ctrl_size>::control(const mjData* d, const std::vector<ctrl_vector>& ddp_ctrl, const std::vector<ctrl_matrix>& ddp_variance)
{
    ctrl_vector instant_control;

    fill_state_vector(m_d_cp, m_state.front());
    std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);
    m_normX_cholesk.setMean(m_params.pi_ctrl_mean);
//    m_normX_cholesk.setCovar(m_params.ctrl_variance);
    ctrl_vector benchmark; benchmark.setConstant(m_m->actuator_ctrlrange[1]);
//    std::cout << "[MEAN]: " << m_params.pi_ctrl_mean << "\n";
//    std::cout << "[COVAR]: " << m_params.ddp_variance(1,1) << "\n";

    for (auto sample = 0; sample < m_params.m_k_samples; ++sample)
    {
        auto func = [](const mjData* data, const mjModel *model){
            std::array<int, 3> joint_list {{0, 1, 2}};
            for(auto i = 0; i < data->ncon; ++i)
            {
                bool check_1 = (std::find(joint_list.begin(), joint_list.end(), model->geom_bodyid[data->contact[i].geom1]) != joint_list.end());
                bool check_2 = (std::find(joint_list.begin(), joint_list.end(), model->geom_bodyid[data->contact[i].geom2]) != joint_list.end());

                if (check_1 != check_2)
                    return true;
            }
            return false;
        };

        // Variance not adapted in this case
        // dU ~ N(mean, variance). Generate samples = to the number of time steps
        copy_data(m_m, d, m_d_cp);
        const auto samples = m_normX_cholesk.samples_vector();
        for(auto time = 0; time < m_params.m_sim_time - 1; ++time)
        {
            m_params.ddp_variance = ddp_variance[time];

            m_ctrl_samples_time.block(sample, time*n_ctrl, 1, n_ctrl) =
                    samples.block(0, time, n_ctrl, 1).transpose().eval();

            ctrl_vector instant_pert = m_ctrl_samples_time.block(
                    sample, time*n_ctrl, 1, n_ctrl
                    ).transpose().eval();

            m_delta_control[time][sample] = instant_pert;
            instant_control = m_control[time] + instant_pert;

            // Forward simulate controls
            set_control_data(m_d_cp, instant_control);
            mj_step(m_m, m_d_cp);
            fill_state_vector(m_d_cp, m_state[time + 1]);

            // Compute cost-to-go of the controls
//            if (func(m_d_cp, m_m))
//                std::cout << func(m_d_cp, m_m) << "\n";

             m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] +
                    m_cost_func(m_state[time + 1],m_control[time], instant_pert, ddp_ctrl[time]) + func(m_d_cp, m_m) * 1000;
        }
        m_ctrl_samples_time.block(sample, (m_params.m_sim_time-1)*n_ctrl, 1, n_ctrl) =
                samples.block(0, m_params.m_sim_time-1, n_ctrl, 1).transpose().eval();

        m_delta_cost_to_go[sample] = m_delta_cost_to_go[sample] + m_cost_func.terminal_cost(m_state.back());
        traj_cost += std::accumulate(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0.0
                )/m_delta_cost_to_go.size();
    }

    traj_cost /= m_params.m_k_samples;
    compute_control_trajectory();
}


template<int state_size, int ctrl_size>
MPPIDDP<state_size, ctrl_size>::~MPPIDDP()
{
    mj_deleteData(m_d_cp);
}

template class MPPIDDP<n_jpos + n_jvel, n_ctrl>;