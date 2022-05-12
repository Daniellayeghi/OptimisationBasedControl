
#ifndef OPTCONTROL_MUJOCO_PAR_MPPI_DDP_H
#define OPTCONTROL_MUJOCO_PAR_MPPI_DDP_H

#include <utility>
#include <vector>
#include <iostream>
#include "mujoco.h"
#include "generic_control.h"
#include "../parameters/simulation_params.h"
#include "../utilities/eigen_norm_dist.h"
#include "../utilities/generic_utils.h"
#include "Eigen/Core"


using namespace SimulationParameters;
using namespace GenericUtils;

using namespace SimulationParameters;
using namespace GenericUtils;

struct MPPIDDPParamsPar{
    const unsigned int m_k_samples  = 0;
    const int m_sim_time   = 0;
    const float m_lambda   = 0;
    const float importance = 0;
    const int m_scale      = 0;
    const int iteration    = 0;
    const double& ddp_cov_reg = 1;
    const CtrlVector pi_ctrl_mean;
    const CtrlMatrix ddp_variance;
    const CtrlMatrix ctrl_variance;
    const FastPair<std::vector<CtrlVector>&, std::vector<CtrlMatrix>&> m_ddp_args;
    const int m_seed = 1;
    const std::function<double(const mjData* data, const mjModel *model)>&  m_importance_reg =
            [&](const mjData* data=nullptr, const mjModel *model=nullptr){return 1.0;};
    const bool m_grav_comp = false;
};

class QRCostDDPPar
{
public:
    QRCostDDPPar(const MPPIDDPParamsPar &params,
                 const std::function<double(const StateVector&, const CtrlVector&, const mjData* data, const mjModel *model)> running_cost,
                 const std::function<double(const StateVector&, const mjData* data, const mjModel *model)> terminal_cost
                 ):
            m_params(params),
            m_running_cost(running_cost),
            m_terminal_cost(terminal_cost)
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
        const auto importance = m_params.importance * m_params.m_importance_reg(data, model);
        CtrlVector new_control = control + delta_control;
        double ddp_bias = (
                (new_control - ddp_mean_control).transpose() * ddp_covariance_inv *  (new_control - ddp_mean_control)
                          )(0, 0) * m_params.importance;

        double passive_bias = (
                new_control.transpose() * m_ctrl_variance_inv * new_control
                              )(0, 0) * (- m_params.importance);

        double common_bias = (
                control.transpose() * m_ctrl_variance_inv * control +
                2 * new_control.transpose() * m_ctrl_variance_inv * control
        )(0, 0);


        const double cost_power = 1;
        return 0.5 * (ddp_bias + passive_bias + common_bias) * m_params.m_lambda + m_running_cost(state, delta_control, data, model) * cost_power;
    }

    const MPPIDDPParamsPar& m_params;
    CtrlMatrix m_ctrl_variance_inv;

public:

    double compute_trajectory_cost(const std::vector<CtrlVector>& ctrl, std::vector<StateVector>& state, const mjData *d, const mjModel *m) const
    {
        auto total_cost = 0.0;
        for(auto time = 0; time < m_params.m_sim_time-1; ++time)
        {
            total_cost += m_running_cost(state[time], ctrl[time], d, m);
        }
        total_cost += m_terminal_cost(state.back(), d, m);
        return total_cost;
    }

    void compute_state_value(const std::vector<CtrlVector>& u_traj, const std::vector<StateVector>& x_traj,
                             FastPair<std::vector<StateVector>, std::vector<double>>& state_value_vec,
                             const mjData* d, const mjModel *m) const
    {
        for(int t = 0; t < m_params.m_sim_time; ++t)
        {
            state_value_vec.first[t] = x_traj[t];
            state_value_vec.second[t] = compute_running_cost(x_traj, u_traj, d, m , t);
        }

        state_value_vec.first.back() = x_traj.back();
        state_value_vec.second.back() = m_terminal_cost(x_traj.back(), d, m);
    }

    const std::function<double(const StateVector&, const CtrlVector&, const mjData* data, const mjModel *model)> m_running_cost;
    const std::function<double(const StateVector&, const mjData* data, const mjModel *model)> m_terminal_cost;

private:

    double compute_running_cost(const std::vector<StateVector>& x_traj,
                                const std::vector<CtrlVector>& u_traj,
                                const mjData* d, const mjModel *m, const int time) const
    {
        double run_cost = 0;
         for (int t = time; t < u_traj.size(); ++t)
         {
             run_cost += m_running_cost(x_traj[t], u_traj[t], d, m);
         }

        return run_cost;
    }
};


class MPPIDDPPar : public BaseController<MPPIDDPPar> {
    friend class BaseController<MPPIDDPPar>;

public:
    // Functions
    explicit MPPIDDPPar(const mjModel *m, QRCostDDPPar &cost, MPPIDDPParamsPar &params);
    void control(const mjData *d, bool skip = false) override;

    ~MPPIDDPPar() = default;

private:
    void compute_cov_from_hess(const std::vector<CtrlMatrix>& ddp_variance);
    void rollout_trajectories(const mjData *d);
    void fill_ctrl_samples();
    void exponentiate_costs(double min_cost);
    void convert_costs_to_is_weight();
    double compute_normalisation_constant();
    void weight_samples_ctrl_traj();
    void perturb_ctrl_traj();


    // Data
    std::vector<std::vector<double>> m_padded_cst;
    std::vector<Eigen::Matrix<double, -1, -1>> m_sample_ctrl_traj;
    std::vector<Eigen::EigenMultivariateNormal<double>> m_dist_gens;
    std::vector<mjData *> m_thread_mjdata;
    std::vector<CtrlMatrix> m_ddp_cov_inv_vec;
    struct m_ThreadData{
        StateVector current = StateVector::Zero(), next = StateVector::Zero();
        CtrlVector instant_ctrl = CtrlVector::Zero();
    };

    unsigned int m_per_thread_sample;
    const mjModel* m_m;
    MPPIDDPParamsPar& m_params;
    const QRCostDDPPar& m_cost_func;
};

#endif //OPTCONTROL_MUJOCO_PAR_MPPI_DDP_H