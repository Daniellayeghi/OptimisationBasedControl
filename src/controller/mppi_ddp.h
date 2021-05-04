
#define NEW 1
#define OLD 0

#if NEW

#ifndef OPTCONTROL_MUJOCO_MPPI_DDP_H
#define OPTCONTROL_MUJOCO_MPPI_DDP_H

#include "mujoco.h"
#include "../parameters/simulation_params.h"
#include "Eigen/Core"
#include "../utilities/eigen_norm_dist.h"
#include <vector>
#include <iostream>


template<int ctrl_size>
struct MPPIDDPParams{
    int m_k_samples  = 0;
    int m_sim_time   = 0;
    float m_lambda   = 0;
    float importance = 0;
    int m_scale      = 0;
    Eigen::Matrix<double, ctrl_size, 1> pi_ctrl_mean;
    Eigen::Matrix<double, ctrl_size, ctrl_size> ddp_variance;
    Eigen::Matrix<double, ctrl_size, ctrl_size> ctrl_variance;
};


template<int state_size, int ctrl_size>
class QRCostDDP
{
    using q_matrix    = Eigen::Matrix<double, state_size, state_size>;
    using r_matrix    = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;

public:
    QRCostDDP(const q_matrix& t_state_reg,
              const q_matrix& r_state_reg,
              const r_matrix& control_reg,
              const state_vector& state_goal,
              const ctrl_vector& ctrl_goal,
              const MPPIDDPParams<ctrl_size> &params
    ):
            m_params(params),
            m_state_goal(state_goal),
            m_control_goal(ctrl_goal),
            m_control_reg(control_reg),
            m_t_state_reg(t_state_reg),
            m_r_state_reg(r_state_reg)
    {
        m_ctrl_variance_inv = m_params.ctrl_variance.inverse();
        m_ddp_variance_inv = m_params.ddp_variance.inverse();
    }

    double terminal_cost(const state_vector& state) const
    {
        state_vector state_error  = m_state_goal - state;
        return state_error.transpose() * m_t_state_reg * state_error;
//        double result = 0;
//        result += 500000 * std::pow(1 - std::cos(state(2, 0)), 2);
//        result += 5000 * std::pow(state(5, 0), 2);
//        return result;
    }


    double operator()(const state_vector& state,
                      const ctrl_vector& control,
                      const ctrl_vector& delta_control,
                      const ctrl_vector& ddp_mean_control)
    {
        m_ddp_variance_inv = m_params.ddp_variance.inverse();
        m_ctrl_variance_inv = m_params.ctrl_variance.inverse();

//        std::cout << "ddp_var_inv " << m_ddp_variance_inv << "\n";
//        std::cout << "ctrl_var_inv  " << m_ctrl_variance_inv << "\n";

        ctrl_vector new_control = control + delta_control;

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
        return (ddp_bias + pi_bias) * m_params.m_lambda + bounded_state_error(state, delta_control) * cost_power;
    }



    double bounded_state_error([[maybe_unused]]const state_vector& state, const ctrl_vector& control) const
    {
//        ctrl_error.transpose() * m_control_reg * ctrl_error
        state_vector state_error = m_state_goal - state;
        ctrl_vector ctrl_error = m_control_goal - control;
        return (ctrl_error.transpose() * m_control_reg * ctrl_error+ state.transpose() * m_r_state_reg * state)(0, 0);
    }

private:

    const MPPIDDPParams<ctrl_size>& m_params;
    state_vector m_state_goal;
    ctrl_vector  m_control_goal;
    r_matrix m_control_reg;
    r_matrix m_ddp_variance_inv;
    r_matrix m_ctrl_variance_inv;
    q_matrix m_t_state_reg;
    q_matrix m_r_state_reg;
};


template<int state_size, int ctrl_size>
class MPPIDDP
{
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;
    using ctrl_matrix  = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_matrix = Eigen::Matrix<double, state_size, state_size>;

public:
    explicit MPPIDDP(const mjModel* m, QRCostDDP<state_size, ctrl_size>& cost, MPPIDDPParams<ctrl_size>& params);

    ~MPPIDDP();

    void control(const mjData* d, const std::vector<ctrl_vector>& ddp_ctrl, const std::vector<ctrl_matrix>& ddp_variance);

    ctrl_vector _cached_control;
    std::vector<ctrl_vector> m_control;
    double traj_cost{};

private:

    void compute_control_trajectory();
    std::pair<ctrl_vector, ctrl_matrix> total_entropy(int time, double min_cost) const;

    MPPIDDPParams<ctrl_size>& m_params;
    std::vector<ctrl_matrix> covariance;
    QRCostDDP<state_size, ctrl_size>& m_cost_func;

    //  Data
    std::vector<state_vector> m_state;
    std::vector<double> m_delta_cost_to_go;
    [[maybe_unused]] std::vector<mjtNum> m_cost;
    std::vector<std::vector<double>> m_cost_to_go_sample_time;
    std::vector<std::vector<ctrl_vector>> m_delta_control;
    // Cache friendly structure [ctrl1_1, ctrl2_1, ctrl1_2, ctrl2_2, ...]
    // Each row contains one ctrl trajectory sample the size of the sim_time * n_ctrl
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_ctrl_samples_time;

    // Models
    const mjModel* m_m;
    mjData*  m_d_cp = nullptr;
    mjData*  m_time_d_cp = nullptr;
    Eigen::EigenMultivariateNormal<double> m_normX_cholesk;
};

#endif //OPTCONTROL_MUJOCO_MPPI_DDP_H
#endif