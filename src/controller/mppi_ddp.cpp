
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
                                        MPPIDDPParams& params):
        m_params(params),
        m_cost_func(cost),
        m_m(m),
        m_normX_cholesk(m_params.pi_ctrl_mean, params.ctrl_variance, m_params.m_sim_time, true)

{
    m_d_cp = mj_makeData(m_m);
    _cached_control = CtrlVector::Zero();

    m_state_new.assign(m_params.m_sim_time + 1, Eigen::Matrix<double, state_size, 1>::Zero());
    m_control.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    m_control_filtered.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    m_control_new.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    m_control_cp.assign(m_params.m_sim_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());

    m_ddp_cov_vec.assign(m_params.m_sim_time, CtrlMatrix::Identity());
    m_delta_cost_to_go.assign(m_params.m_k_samples,0);
    m_ctrl_samp_time.resize(m_params.m_k_samples, m_params.m_sim_time);
    m_ctrl_samples_time.resize(m_params.m_k_samples, m_params.m_sim_time * n_ctrl);
    m_cost_to_go_sample_time.assign(m_params.m_k_samples, std::vector<double>(m_params.m_sim_time, 0));
}


template<int state_size, int ctrl_size>
FastPair<CtrlVector, CtrlMatrix>
MPPIDDP<state_size, ctrl_size>::total_entropy(const int time, const double min_cost, const double normaliser)
{

    // Computing the new covariance is taken from:
    // "Path Integral Policy Improvement with Covariance Matrix Adaptation"
    CtrlVector numerator_mean = CtrlVector::Zero();
    CtrlMatrix numerator_cov  = CtrlMatrix::Zero();

    // Swap the order of samples and time
    auto ctrl_time_samples = m_ctrl_samples_time.transpose();
    auto numerator_weight = 0.0;
    for (unsigned long col = 0; col < m_delta_cost_to_go.size(); ++col)
    {

        numerator_weight = std::exp(-(1 / m_params.m_lambda) * (m_delta_cost_to_go[col] - min_cost)) /normaliser;
        auto ctrl_sample = ctrl_time_samples.block(time*ctrl_size, col, ctrl_size, 1);
        numerator_mean += (numerator_weight * ctrl_sample);
        numerator_cov += (
                numerator_weight * (ctrl_sample - m_params.pi_ctrl_mean) *
                (ctrl_sample - m_params.pi_ctrl_mean).transpose()
                );
    }
    return {numerator_mean, numerator_cov/normaliser};
}


template <int state_size, int ctrl_size>
bool MPPIDDP<state_size, ctrl_size>::MPPIDDP::accepted_trajectory()
{
    MujocoUtils::rollout_dynamics(m_control_new, m_state_new, m_d_cp, m_m);
    auto total_cost = compute_trajectory_cost(m_control_new, m_state_new);
    std::cout << total_cost << " " << m_prev_cost << "\n";
    if(total_cost < m_prev_cost)
    {
        m_prev_cost = total_cost;
        return true;
    }
    return false;
}


template <int state_size, int ctrl_size>
double MPPIDDP<state_size, ctrl_size>::MPPIDDP::compute_trajectory_cost(const std::vector<CtrlVector>& ctrl, std::vector<StateVector>& state)
{
    return m_cost_func.compute_trajectory_cost(ctrl, state, m_d_cp, m_m);
}


template <int state_size, int ctrl_size>
FastPair<CtrlVector, CtrlMatrix> MPPIDDP<state_size, ctrl_size>::MPPIDDP::compute_control_trajectory()
{
    const auto min_cost = std::min_element(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end());
    double temp_mean_denomenator = 0;
    CtrlVector new_mean_numerator = CtrlVector::Zero(); CtrlMatrix new_cov_numerator = CtrlMatrix::Zero();
    Eigen::Matrix<double, n_ctrl, n_ctrl> new_variance; new_variance.setZero();

    // Compute the normalisation constant
    double normaliser =  0;
    for (auto& sample_cost: m_delta_cost_to_go)
    {
        auto cost_diff = sample_cost - *min_cost;
        normaliser += (std::exp(-(1 / m_params.m_lambda) * (cost_diff)));
    }

    // Compute wighted samples
    for (auto time = 0; time < m_params.m_sim_time; ++time)
    {
        const auto [ctrl_pert, ctrl_variance] = (total_entropy(time, *min_cost, normaliser));
        m_control[time] += ctrl_pert;

        // Temporal average for the mean and covariance
        new_mean_numerator += (ctrl_pert * ((m_params.m_sim_time - 1) - time));
        temp_mean_denomenator += ((m_params.m_sim_time - 1) - time);
        new_variance += (ctrl_variance * ((m_params.m_sim_time - 1) - time));
    }

//    sg_filter(m_control, m_control_filtered);
//    m_control = m_control_filtered;
    return {new_mean_numerator / temp_mean_denomenator, new_variance / temp_mean_denomenator};
}


template<int state_size, int ctrl_size>
void MPPIDDP<state_size, ctrl_size>::prepare_control_mpc()
{
    _cached_control = m_control.front();
    m_control_cp = m_control;
    std::rotate(m_control.begin(), m_control.begin() + 1, m_control.end());
    m_control.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
}

template<int state_size, int ctrl_size>
void MPPIDDP<state_size, ctrl_size>::regularise_ddp_variance(std::vector<CtrlMatrix>& ddp_variance)
{
    for (auto elem = 0; elem < m_params.m_sim_time; ++elem)
    {
        m_ddp_cov_vec[elem] = ddp_variance[elem] / m_params.ddp_cov_reg;
    }
}



template<int state_size, int ctrl_size>
void MPPIDDP<state_size, ctrl_size>::control(const mjData* d, const std::vector<CtrlVector>& ddp_ctrl, std::vector<CtrlMatrix>& ddp_variance) {

    // TODO: compute the previous trajectory cost here with the new state then compare to the new one
    regularise_ddp_variance(ddp_variance);
    for (auto iteration = 0; iteration < m_params.iteration; ++iteration) {
        CtrlVector instant_control;
        fill_state_vector(d, m_state_new.front(), m_m);
        std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);
        m_normX_cholesk.setMean(m_params.pi_ctrl_mean);
//        m_normX_cholesk.setCovar(m_params.ctrl_variance);
//        std::cout << "[MEAN]: " << m_params.pi_ctrl_mean << "\n";

/*
 *      Update distribution params.
        m_normX_cholesk.setCovar(m_params.ctrl_variance);
        std::cout << "[COVAR]: " << m_params.ddp_variance << "\n";
*/
        for (auto sample = 0; sample < m_params.m_k_samples; ++sample) {
            // Variance not adapted in this case
            // dU ~ N(mean, variance). Generate samples = to the number of time steps
            copy_data(m_m, d, m_d_cp);
            const auto samples = m_normX_cholesk.samples_vector();
            for (auto time = 0; time < m_params.m_sim_time - 1; ++time) {
                // Set sampled perturbation
                auto pert_sample = samples.block(0, time, n_ctrl, 1).transpose().eval();
                m_ctrl_samples_time.block(sample, time * n_ctrl, 1, n_ctrl) = pert_sample;
                CtrlVector instant_pert = pert_sample.transpose().eval();
                instant_control = m_control[time] + instant_pert;
                // Forward simulate controls and compute running costl
                MujocoUtils::apply_ctrl_update_state(instant_control, m_state_new[time + 1], m_d_cp, m_m);
                m_delta_cost_to_go[sample] += m_cost_func(
                        m_state_new[time + 1], m_control[time], instant_pert, ddp_ctrl[time], m_ddp_cov_vec[time], m_d_cp, m_m
                );
            }

            // Set final pert sample
            auto final_sample = samples.block(0, m_params.m_sim_time - 1, n_ctrl, 1).transpose().eval();
            m_ctrl_samples_time.block(sample, (m_params.m_sim_time - 1) * n_ctrl, 1, n_ctrl) = final_sample;

            // Apply final sample
            instant_control = m_control.back() + final_sample.transpose().eval();
            MujocoUtils::apply_ctrl_update_state(instant_control, m_state_new.back(), m_d_cp, m_m);

            // Compute terminal cost
            m_delta_cost_to_go[sample] =
                    m_delta_cost_to_go[sample] + m_cost_func.m_terminal_cost(m_state_new.back(), m_d_cp, m_m);
            traj_cost += std::accumulate(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0.0) /
                         m_delta_cost_to_go.size();
        }

        traj_cost /= m_params.m_k_samples;
        const auto [new_mean, new_variance] = compute_control_trajectory();
        m_params.pi_ctrl_mean = new_mean;
//        m_params.ctrl_variance = new_variance + CtrlMatrix::Identity() * 0.0001;
    }
    prepare_control_mpc();
}


template<int state_size, int ctrl_size>
MPPIDDP<state_size, ctrl_size>::~MPPIDDP()
{
    mj_deleteData(m_d_cp);
}

template class MPPIDDP<state_size, n_ctrl>;