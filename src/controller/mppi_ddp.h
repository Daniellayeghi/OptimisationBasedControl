
#ifndef OPTCONTROL_MUJOCO_MPPI_DDP_H
#define OPTCONTROL_MUJOCO_MPPI_DDP_H

#include "mujoco.h"
#include "generic_control.h"
#include "../parameters/simulation_params.h"
#include "../utilities/eigen_norm_dist_2.h"
#include "../utilities/generic_utils.h"
#include "Eigen/Core"
#include <utility>
#include <vector>
#include <iostream>


using namespace SimulationParameters;
using namespace GenericUtils;

struct MPPIDDPParams{
    const unsigned int m_k_samples  = 0;
    const int m_sim_time   = 0;
    const float m_lambda   = 0;
    const float importance = 0;
    const int m_scale      = 0;
    const int iteration    = 0;
    const double ddp_cov_reg = 1;
    const CtrlVector pi_ctrl_mean;
    const CtrlMatrix ddp_variance;
    const CtrlMatrix ctrl_variance;
    const FastPair<std::vector<CtrlVector>&, std::vector<CtrlMatrix>&> m_ddp_args;
    const unsigned int m_seed = 1;
};

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
    }

    double operator()(const StateVector& state,
                      const CtrlVector& control,
                      const CtrlVector& delta_control,
                      const CtrlVector& ddp_mean_control,
                      const CtrlMatrix& ddp_covariance_inv,
                      const mjData* data, const mjModel *model) const
    {
        CtrlVector new_control = control + delta_control;
        double ddp_bias = (
                (new_control - ddp_mean_control).transpose() * ddp_covariance_inv *  (new_control - ddp_mean_control)
                )(0, 0) * m_params.importance;

        double passive_bias = (
                new_control.transpose() * ddp_covariance_inv * new_control
                )(0, 0) * (1 - m_params.importance);

        double common_bias = (
                (new_control - control).transpose() * m_ctrl_variance_inv * (new_control - control)
                )(0, 0);

        const double cost_power = 1;
        return 0.5 * (ddp_bias + passive_bias + common_bias) * m_params.m_lambda + m_running_cost(state, delta_control, data, model) * cost_power;
    }

    const MPPIDDPParams& m_params;
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


class MPPIDDP : public BaseController<MPPIDDP>
{
    friend class BaseController<MPPIDDP>;
public:
    explicit MPPIDDP(const mjModel* m, QRCostDDP& cost, MPPIDDPParams& params);

    ~MPPIDDP();

    void control(const mjData* d, bool skip = false) override;
private:

    bool accepted_trajectory();
    void prepare_control_mpc(bool skip = false);
    FastPair<CtrlVector, CtrlMatrix> compute_control_trajectory();
    void regularise_ddp_variance(std::vector<CtrlMatrix>& ddp_variance);
    FastPair<CtrlVector, CtrlMatrix> total_entropy(int time, double min_cost, double normaliser);
    double compute_trajectory_cost(const std::vector<CtrlVector>& ctrl, std::vector<StateVector>& state);

    MPPIDDPParams& m_params;
    QRCostDDP& m_cost_func;
    std::vector<double> m_delta_cost_to_go;
    std::vector<CtrlVector> m_control_filtered;
    [[maybe_unused]] std::vector<mjtNum> m_cost;
    std::vector<CtrlMatrix> m_ddp_cov_inv_vec;
    // Cache friendly structure [ctrl1_1, ctrl2_1, ctrl1_2, ctrl2_2, ...]
    // Each row contains one ctrl trajectory sample the size of the sim_time * n_ctrl
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_ctrl_samples_time;

    double traj_cost = 0;
    double m_prev_cost = 0;

    const mjModel* m_m;
    mjData*  m_d_cp = nullptr;
    Eigen::EigenMultivariateNormal2<double> m_normX_cholesk;
};

#endif //OPTCONTROL_MUJOCO_MPPI_DDP_H
