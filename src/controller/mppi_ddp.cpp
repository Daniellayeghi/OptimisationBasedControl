
#include "mppi_ddp.h"
#include "../../src/utilities/mujoco_utils.h"
#include <iostream>

using namespace SimulationParameters;
using namespace MujocoUtils;

MPPIDDP::MPPIDDP(const mjModel* m, QRCostDDP& cost, MPPIDDPParams& params):
        m_params(params),
        m_cost_func(cost),
        m_m(m),
        m_normX_cholesk(m_params.pi_ctrl_mean, params.ctrl_variance, m_params.m_sim_time, true,  m_params.m_seed)
{
    m_d_cp = mj_makeData(m_m);
    cached_control = CtrlVector::Zero();

    m_x_traj.assign(m_params.m_sim_time + 1, StateVector::Zero());
    m_u_traj.assign(m_params.m_sim_time, CtrlVector::Zero());
    m_control_filtered.assign(m_params.m_sim_time, CtrlVector::Zero());
    m_u_traj_new.assign(m_params.m_sim_time, CtrlVector ::Zero());
    m_u_traj_cp.assign(m_params.m_sim_time, CtrlVector::Zero());

    m_ddp_cov_inv_vec.assign(m_params.m_sim_time, CtrlMatrix::Identity());
    m_delta_cost_to_go.assign(m_params.m_k_samples,0);
    m_ctrl_samples_time.resize(m_params.m_k_samples, m_params.m_sim_time * n_ctrl);
}


FastPair<CtrlVector, CtrlMatrix>
MPPIDDP::total_entropy(const int time, const double min_cost, const double normaliser)
{
    constexpr const size_t ctrl_rows = CtrlVector::RowsAtCompileTime;

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
    auto ctrl_sample = ctrl_time_samples.block(time*ctrl_rows, col, ctrl_rows, 1);
        numerator_mean += (numerator_weight * ctrl_sample);
        numerator_cov += (
                numerator_weight * (ctrl_sample - m_params.pi_ctrl_mean) *
                (ctrl_sample - m_params.pi_ctrl_mean).transpose()
                );
    }
    return {numerator_mean, numerator_cov/normaliser};
}


bool MPPIDDP::MPPIDDP::accepted_trajectory()
{
    MujocoUtils::rollout_dynamics(m_u_traj_new, m_x_traj, m_d_cp, m_m);
    auto total_cost = compute_trajectory_cost(m_u_traj_new, m_x_traj);
    std::cout << total_cost << " " << m_prev_cost << "\n";
    if(total_cost < m_prev_cost)
    {
        m_prev_cost = total_cost;
        return true;
    }
    return false;
}


double MPPIDDP::MPPIDDP::compute_trajectory_cost(const std::vector<CtrlVector>& ctrl, std::vector<StateVector>& state)
{
    return m_cost_func.compute_trajectory_cost(ctrl, state, m_d_cp, m_m);
}


FastPair<CtrlVector, CtrlMatrix> MPPIDDP::compute_control_trajectory()
{
    const auto min_cost = std::min_element(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end());
    double temp_mean_denomenator = 0;
    CtrlVector new_mean_numerator = CtrlVector::Zero();
    CtrlMatrix new_cov_numerator = CtrlMatrix::Zero();
    CtrlMatrix new_variance = CtrlMatrix::Zero();

    // Compute the normalisation constant
    double normaliser =  0;
    for (auto& sample_cost: m_delta_cost_to_go)
    {
        auto cost_diff = sample_cost - *min_cost;
        auto val = (std::exp(-(1 / m_params.m_lambda) * (cost_diff)));
        normaliser += (std::exp(-(1 / m_params.m_lambda) * (cost_diff)));
    }

    // Compute wighted samples
    for (auto time = 0; time < m_params.m_sim_time; ++time)
    {
        const auto [ctrl_pert, ctrl_variance] = (total_entropy(time, *min_cost, normaliser));
        m_u_traj[time] += ctrl_pert;

        // Temporal average for the mean and covariance
        new_mean_numerator += (ctrl_pert * ((m_params.m_sim_time - 1) - time));
        temp_mean_denomenator += ((m_params.m_sim_time - 1) - time);
        new_variance += (ctrl_variance * ((m_params.m_sim_time - 1) - time));
    }
//
//    sg_filter(m_control, m_control_filtered);
//    m_control = m_control_filtered;
    return {new_mean_numerator / temp_mean_denomenator, new_variance / temp_mean_denomenator};
}


void MPPIDDP::prepare_control_mpc(const bool skip)
{
    if (not skip)
    {
        MujocoUtils::rollout_dynamics(m_u_traj_new, m_x_traj, m_d_cp, m_m);
        traj_cost = compute_trajectory_cost(m_u_traj_new, m_x_traj);
        cached_control = m_u_traj.front();
        m_u_traj_cp = m_u_traj;
    }
    std::rotate(m_u_traj.begin(), m_u_traj.begin() + 1, m_u_traj.end());
    m_u_traj.back() = CtrlVector::Zero();
}


void MPPIDDP::regularise_ddp_variance(std::vector<CtrlMatrix>& ddp_variance)
{
    for (auto elem = 0; elem < m_params.m_sim_time; ++elem)
    {
        m_ddp_cov_inv_vec[elem] = (ddp_variance[elem] / m_params.ddp_cov_reg).llt().solve(CtrlMatrix::Identity());
    }
}


void MPPIDDP::control(const mjData* d, const bool skip)
{
    // TODO: compute the previous trajectory cost here with the new state then compare to the new one
    if (not skip)
    {
        regularise_ddp_variance(m_params.m_ddp_args.second);
        for (auto iteration = 0; iteration < m_params.iteration; ++iteration)
        {
            CtrlVector instant_control;
            fill_state_vector(d, m_x_traj.front(), m_m);
            std::fill(m_delta_cost_to_go.begin(), m_delta_cost_to_go.end(), 0);
            for (auto sample = 0; sample < m_params.m_k_samples; ++sample) {
                // Variance not adapted in this case
                // dU ~ N(mean, variance). Generate samples = to the number of time steps
                copy_data(m_m, d, m_d_cp);
                m_normX_cholesk.samples_fill(m_ctrl_samples_time.row(sample));
                for (auto time = 0; time < m_params.m_sim_time-1 ; ++time){
                    // Set sampled perturbation
                    const CtrlVector& pert_sample = m_ctrl_samples_time.block(sample, time*n_ctrl, n_ctrl, 1);
                    instant_control = m_u_traj[time] + pert_sample;
                    // Forward simulate controls and compute running costl
                    MujocoUtils::apply_ctrl_update_state(instant_control, m_x_traj[time + 1], m_d_cp, m_m);
                    m_delta_cost_to_go[sample] +=m_cost_func(
                            m_x_traj[time + 1], m_u_traj[time],
                            pert_sample, m_params.m_ddp_args.first[time],
                            m_ddp_cov_inv_vec[time],
                            m_d_cp, m_m);
                }
//                printf("sample %d cost %f \n",sample, m_delta_cost_to_go[sample]);
                // Set final pert sample
                const CtrlVector& final_sample = m_ctrl_samples_time.block(
                        sample, m_params.m_sim_time - 1, n_ctrl, 1
                        );

                // Apply final sample
                instant_control = m_u_traj.back() + final_sample;
                MujocoUtils::apply_ctrl_update_state(instant_control, m_x_traj.back(), m_d_cp, m_m);

                // Compute terminal cost
                m_delta_cost_to_go[sample] += m_cost_func.m_terminal_cost(m_x_traj.back(), m_d_cp, m_m);

            }
            std::cout << m_ctrl_samples_time << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            auto k = 1; std::cin >> k;
            const auto[new_mean, new_variance] = compute_control_trajectory();
//            m_params.pi_ctrl_mean = new_mean;
//        m_params.ctrl_variance = new_variance + CtrlMatrix::Identity() * 0.0001;
        }
    }
    prepare_control_mpc();
}


MPPIDDP::~MPPIDDP()
{
    mj_deleteData(m_d_cp);
}
