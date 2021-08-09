
#ifndef OPTCONTROL_MUJOCO_MPPI_DDP_H
#define OPTCONTROL_MUJOCO_MPPI_DDP_H

#include "mujoco.h"
#include "../parameters/simulation_params.h"
#include "../utilities/eigen_norm_dist.h"
#include "../utilities/generic_utils.h"
#include "Eigen/Core"
#include <utility>
#include <vector>
#include <iostream>


using namespace SimulationParameters;
using namespace GenericUtils;

struct MPPIDDPParams{
    int m_k_samples  = 0;
    int m_sim_time   = 0;
    float m_lambda   = 0;
    float importance = 0;
    int m_scale      = 0;
    int iteration    = 0;
    double ddp_cov_reg = 1;
    CtrlVector pi_ctrl_mean;
    CtrlMatrix ddp_variance;
    CtrlMatrix ctrl_variance;
};


template<int state_size, int ctrl_size>
class QRCostDDP
{
public:
    QRCostDDP(const MPPIDDPParams &params,
              std::function<double(const StateVector&, const CtrlVector&, const mjData* data, const mjModel *model)> running_cost,
              std::function<double(const StateVector&, const mjData* data, const mjModel *model)> terminal_cost
    ):
            m_params(params),
            m_running_cost(std::move(running_cost)),
            m_terminal_cost(std::move(terminal_cost))
    {
        m_ctrl_variance_inv = m_params.ctrl_variance.inverse();
        m_ddp_variance_inv = m_params.ddp_variance.inverse();
    }

    double operator()(const StateVector& state,
                      const CtrlVector& control,
                      const CtrlVector& delta_control,
                      const CtrlVector& ddp_mean_control,
                      const CtrlMatrix& ddp_covariance,
                      const mjData* data, const mjModel *model)
    {

        m_ddp_variance_inv = ddp_covariance.llt().solve(CtrlMatrix::Identity());

        CtrlVector new_control = control + delta_control;

        auto ddp_noise_term = (new_control.transpose().eval() * m_ddp_variance_inv * new_control -
                2 * (new_control.transpose().eval() * m_ddp_variance_inv * ddp_mean_control)
                );

        double ddp_bias = (ddp_noise_term + ddp_mean_control.transpose().eval() * m_ddp_variance_inv * ddp_mean_control -
                (new_control.transpose().eval() * m_ctrl_variance_inv * new_control)
                )(0, 0) * m_params.importance;

        double pi_bias = (2 * (new_control.transpose().eval() * m_ctrl_variance_inv * control) -
                control.transpose().eval() * m_ctrl_variance_inv * control
                )(0, 0);

        const double cost_power = 1;
        return -0.5*(ddp_bias + pi_bias) * m_params.m_lambda + m_running_cost(state, delta_control, data, model) * cost_power;
    }

    const MPPIDDPParams& m_params;
    CtrlMatrix m_ddp_variance_inv;
    CtrlMatrix m_ctrl_variance_inv;

public:

    double compute_trajectory_cost(const std::vector<CtrlVector>& ctrl, std::vector<StateVector>& state, mjData *d, const mjModel *m) const
    {
        auto total_cost = 0.0;
        for(auto time = 0; time < m_params.m_sim_time-1; ++time)
        {
            total_cost += m_running_cost(state[time + 1], ctrl[time], d, m);
        }
        total_cost += m_terminal_cost(state.back(), d, m);
        return total_cost;
    }

    const std::function<double(const StateVector&, const CtrlVector&, const mjData* data, const mjModel *model)> m_running_cost;
    const std::function<double(const StateVector&, const mjData* data, const mjModel *model)> m_terminal_cost;
};


template<int state_size, int ctrl_size>
class MPPIDDP
{
public:
    explicit MPPIDDP(const mjModel* m, QRCostDDP<state_size, ctrl_size>& cost, MPPIDDPParams& params);

    ~MPPIDDP();

    void control(const mjData* d, const std::vector<CtrlVector>& ddp_ctrl, std::vector<CtrlMatrix>& ddp_variance);

    CtrlVector _cached_control;
    std::vector<CtrlVector> m_control;
    std::vector<CtrlVector> m_control_new;
    std::vector<CtrlVector> m_control_cp;
    double traj_cost{};

private:

    FastPair<CtrlVector, CtrlMatrix> compute_control_trajectory();
    FastPair<CtrlVector, CtrlMatrix> total_entropy(int time, double min_cost, double normaliser);
    void regularise_ddp_variance(std::vector<CtrlMatrix>& ddp_variance);
    void prepare_control_mpc();
    double compute_trajectory_cost(const std::vector<CtrlVector>& ctrl, std::vector<StateVector>& state);
    bool accepted_trajectory();

    MPPIDDPParams& m_params;
    std::vector<CtrlMatrix> covariance;
    QRCostDDP<state_size, ctrl_size>& m_cost_func;

    //  Data
    std::vector<StateVector> m_state_new;
    std::vector<double> m_delta_cost_to_go;
    [[maybe_unused]] std::vector<mjtNum> m_cost;
    std::vector<std::vector<double>> m_cost_to_go_sample_time;
    std::vector<CtrlMatrix> m_ddp_cov_vec;

    // Cache friendly structure [ctrl1_1, ctrl2_1, ctrl1_2, ctrl2_2, ...]
    // Each row contains one ctrl trajectory sample the size of the sim_time * n_ctrl
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_ctrl_samples_time;
    Eigen::Matrix<CtrlVector, Eigen::Dynamic, Eigen::Dynamic> m_ctrl_samp_time;
    double m_prev_cost = 0;
    const mjModel* m_m;
    mjData*  m_d_cp = nullptr;
    Eigen::EigenMultivariateNormal<double> m_normX_cholesk;
};

#endif //OPTCONTROL_MUJOCO_MPPI_DDP_H
