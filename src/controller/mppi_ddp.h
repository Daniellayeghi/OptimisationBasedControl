
#ifndef OPTCONTROL_MUJOCO_MPPI_DDP_H
#define OPTCONTROL_MUJOCO_MPPI_DDP_H

#include "mujoco.h"
#include "../parameters/simulation_params.h"
#include "Eigen/Core"
#include "../utilities/eigen_norm_dist.h"
#include <utility>
#include <vector>
#include <iostream>


using namespace SimulationParameters;


template<int ctrl_size>
struct MPPIDDPParams{
    int m_k_samples  = 0;
    int m_sim_time   = 0;
    float m_lambda   = 0;
    float importance = 0;
    int m_scale      = 0;
    CtrlVector pi_ctrl_mean;
    CtrlMatrix ddp_variance;
    CtrlMatrix ctrl_variance;
};


template<int state_size, int ctrl_size>
class QRCostDDP
{
public:
    QRCostDDP(const double ddp_variance_reg,
              const MPPIDDPParams<ctrl_size> &params,
              std::function<double(const StateVector&, const CtrlVector&, const mjData* data, const mjModel *model)> running_cost,
              std::function<double(const StateVector&)> terminal_cost
    ):
            m_ddp_variance_reg(ddp_variance_reg),
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
                      const mjData* data, const mjModel *model)
    {
        m_ddp_variance_inv = ((m_params.ddp_variance+CtrlMatrix::Identity()*1e-6)/m_ddp_variance_reg).inverse();
        m_ctrl_variance_inv = m_params.ctrl_variance.inverse();

        CtrlVector new_control = control + delta_control;

        auto ddp_noise_term = (new_control.transpose() * m_ddp_variance_inv * new_control -
                2 * new_control.transpose() * m_ddp_variance_inv * ddp_mean_control
                ) * 1;

        double ddp_bias = (ddp_noise_term + ddp_mean_control.transpose() * m_ddp_variance_inv * ddp_mean_control -
                (1 + m_params.m_scale) * new_control.transpose() * m_ctrl_variance_inv * new_control
                )(0, 0) * m_params.importance/(m_params.importance + 1);

        double pi_bias = (2 * new_control.transpose() * m_ctrl_variance_inv * control -
                control.transpose() * m_ctrl_variance_inv * control
                )(0, 0);

        const double cost_power = std::pow(1/(m_params.importance + 1), m_params.m_scale);
        return (ddp_bias + pi_bias) * m_params.m_lambda + m_running_cost(state, delta_control, data, model) * cost_power;
    }


private:
    const double m_ddp_variance_reg;
    const MPPIDDPParams<ctrl_size>& m_params;
    CtrlMatrix m_ddp_variance_inv;
    CtrlMatrix m_ctrl_variance_inv;
    const std::function<double(const StateVector&, const CtrlVector&, const mjData* data, const mjModel *model)> m_running_cost;

public:
    const std::function<double(const StateVector&)> m_terminal_cost;
};


template<int state_size, int ctrl_size>
class MPPIDDP
{
public:
    explicit MPPIDDP(const mjModel* m, QRCostDDP<state_size, ctrl_size>& cost, MPPIDDPParams<ctrl_size>& params);

    ~MPPIDDP();

    void control(const mjData* d, const std::vector<CtrlVector>& ddp_ctrl, const std::vector<CtrlMatrix>& ddp_variance);

    CtrlVector _cached_control;
    std::vector<CtrlVector> m_control;
    std::vector<CtrlVector> m_control_cp;
    double traj_cost{};

private:

    void compute_control_trajectory();
    std::pair<CtrlVector, CtrlMatrix> total_entropy(int time, double min_cost) const;

    MPPIDDPParams<ctrl_size>& m_params;
    std::vector<CtrlMatrix> covariance;
    QRCostDDP<state_size, ctrl_size>& m_cost_func;

    //  Data
    std::vector<StateVector> m_state;
    std::vector<double> m_delta_cost_to_go;
    [[maybe_unused]] std::vector<mjtNum> m_cost;
    std::vector<std::vector<double>> m_cost_to_go_sample_time;
    std::vector<std::vector<CtrlVector>> m_delta_control;
    // Cache friendly structure [ctrl1_1, ctrl2_1, ctrl1_2, ctrl2_2, ...]
    // Each row contains one ctrl trajectory sample the size of the sim_time * n_ctrl
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_ctrl_samples_time;
    Eigen::Matrix<CtrlVector, Eigen::Dynamic, Eigen::Dynamic> m_ctrl_samp_time;

    // Models
    const mjModel* m_m;
    mjData*  m_d_cp = nullptr;
    Eigen::EigenMultivariateNormal<double> m_normX_cholesk;
};

#endif //OPTCONTROL_MUJOCO_MPPI_DDP_H
