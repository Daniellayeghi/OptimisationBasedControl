
#ifndef OPTCONTROL_MUJOCO_MPPI_DDP_H
#define OPTCONTROL_MUJOCO_MPPI_DDP_H

#include"mujoco.h"
#include "Eigen/Core"
#include <vector>
#include <iostream>

template<int state_size, int ctrl_size>
class QRCostDDP
{
    using q_matrix    = Eigen::Matrix<double, state_size, state_size>;
    using r_matrix    = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;

public:
    QRCostDDP(const r_matrix & ddp_variance_inv,
           const r_matrix & ctrl_variance_inv,
           const q_matrix & state_reg,
           const state_vector &state_goal,
           const ctrl_vector &ctrl_goal
    ):
            m_ddp_variance_inv(ddp_variance_inv),
            m_ctrl_variance_inv(ctrl_variance_inv),
            m_state_goal(state_goal),
            m_control_goal(ctrl_goal),
            m_state_reg(state_reg)
    {}

    double operator()(state_vector& state,
                      ctrl_vector& control,
                      ctrl_vector& delta_control,
                      ctrl_vector& ddp_mean_control,
                      double lambda, double importance = 0) const
    {
        ctrl_vector new_control = control + delta_control;
        state_vector state_error  = m_state_goal - state;
        ctrl_vector control_error = m_control_goal - control;

        double ddp_bias = (new_control.template transpose() * m_ddp_variance_inv * new_control -
                           2 * new_control.template transpose() * m_ddp_variance_inv * ddp_mean_control +
                           ddp_mean_control.template transpose() * m_ddp_variance_inv * ddp_mean_control +
                           new_control.template transpose() * m_ctrl_variance_inv * new_control)(0, 0) * importance/(importance + 1);

        double pi_bias = (2 * new_control.template transpose() * m_ctrl_variance_inv * control -
                          control.template transpose() * m_ctrl_variance_inv * control)(0, 0);


        std::cout << "ddp: " << ddp_bias << std::endl;
        return (ddp_bias + pi_bias) * lambda;
    }


    double terminal_cost(Eigen::Matrix<double, state_size, 1>& state) const
    {
        state_vector state_error  = m_state_goal - state;
        return state_error.transpose() * m_state_reg * state_error;
    }

private:

    state_vector m_state_goal;
    ctrl_vector  m_control_goal;
    r_matrix m_ddp_variance_inv;
    r_matrix m_ctrl_variance_inv;
    q_matrix m_state_reg;

};

struct MPPIDDPParams
{
    int m_k_samples  = 0;
    int m_sim_time   = 0;
    float m_variance = 0;
    float m_lambda   = 0;
    float importance = 0;
};


template<int state_size, int ctrl_size>
class MPPIDDP
{
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;

public:
    explicit MPPIDDP(const mjModel* m,
                     const QRCostDDP<state_size, ctrl_size>& cost,
                     const MPPIDDPParams& params);

    ~MPPIDDP();

    void control(const mjData* d, std::vector<MPPIDDP<state_size, ctrl_size>::ctrl_vector>& ddp_ctrl);

    ctrl_vector _cached_control;
    double traj_cost{};
private:

    MPPIDDP<state_size, ctrl_size>::ctrl_vector total_entropy(const std::vector<ctrl_vector>& delta_control_samples, double min_cost) const;

    void compute_control_trajectory();

    const MPPIDDPParams& m_params;
    QRCostDDP<state_size, ctrl_size> m_cost_func{};
    std::vector<state_vector> m_state;

    std::vector<ctrl_vector> m_control;
    std::vector<double> m_delta_cost_to_go;

    std::vector<std::vector<ctrl_vector>> m_delta_control;
    const mjModel* m_m;

    mjData*  m_d_cp = nullptr;
    [[maybe_unused]] std::vector<mjtNum> m_cost;
};

#endif //OPTCONTROL_MUJOCO_MPPI_DDP_H
