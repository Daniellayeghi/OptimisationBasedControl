#ifndef OPTCONTROL_MUJOCO_MPPI_H
#define OPTCONTROL_MUJOCO_MPPI_H

#include"mujoco.h"
#include "Eigen/Core"
#include <vector>
#include <iostream>

#if 1
template<int state_size, int ctrl_size>
class QRCost
{
    using q_matrix    = Eigen::Matrix<double, state_size, state_size>;
    using r_matrix    = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;

public:
    QRCost(const r_matrix & R, const q_matrix & Q, const state_vector &state_goal, const ctrl_vector &ctrl_goal)
    : m_Q(Q), m_R(R), m_state_goal(state_goal), m_control_goal(ctrl_goal)
    {}

    double operator()(state_vector& state,
                      ctrl_vector& control,
                      ctrl_vector& delta_control,
                      double variance) const
    {
        state_vector state_error  = m_state_goal - state;
        ctrl_vector control_error = m_control_goal - control;

        double qr_cost = (control_error.template transpose() * m_R * control_error +
                          delta_control.template transpose() * m_R * delta_control +
                          control_error.template transpose() * m_R * control_error * 0.5)(0, 0);

        return (1- 1/variance)/2 + qr_cost + bounded_state_error(state);
    }

private:

    double bounded_state_error(Eigen::Matrix<double, state_size, 1>& state) const
    {
        double result = 0;
        result += 800 * std::pow(1 - std::cos(state(2, 0)), 2);
        result += 50 * std::pow(state(5, 0), 2) * 0.01;
        return result;
    }

    state_vector m_state_goal;
    ctrl_vector  m_control_goal;
    q_matrix m_Q;
    r_matrix m_R;
};
#endif


#if 0
template<int state_size, int ctrl_size>
class QRCost
{
    using q_matrix    = Eigen::Matrix<double, state_size, state_size>;
    using r_matrix    = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;

public:
    QRCost(const r_matrix & R, const q_matrix & Q, const state_vector &state_goal, const ctrl_vector &ctrl_goal)
            : m_Q(Q), m_R(R), m_state_goal(state_goal), m_control_goal(ctrl_goal)
    {}

    double operator()(state_vector& state,
                      ctrl_vector& control,
                      ctrl_vector& delta_control,
                      double variance) const
    {
        state_vector state_error  = m_state_goal - state;
        ctrl_vector control_error = m_control_goal - control;

        double qr_cost = (control_error.template transpose() * m_R * control_error +
                          delta_control.template transpose() * m_R * delta_control +
                          control_error.template transpose() * m_R * control_error * 0.5)(0, 0);

        return (1- 1/variance)/2 + qr_cost + bounded_state_error(state);
    }

private:

    double bounded_state_error(Eigen::Matrix<double, state_size, 1>& state) const
    {
        double result = 0;
        result += 500 * std::pow(1 - std::cos(state(1, 0)), 2);
        result += 800 * std::pow(state(3, 0), 2) * 0.01;
        result += 500 * std::pow(state(0, 0), 2);
        result += 400 * std::pow(state(2, 0), 2) * 0.01;
        return result;
    }

    state_vector m_state_goal;
    ctrl_vector  m_control_goal;
    q_matrix m_Q;
    r_matrix m_R;
};
#endif

struct MPPIParams
{
    int m_k_samples  = 0;
    int m_sim_time   = 0;
    float m_variance = 0;
    float m_lambda   = 0;
};


template<int state_size, int ctrl_size>
class MPPI
{
    using state_vector = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vector  = Eigen::Matrix<double, ctrl_size, 1>;

public:
    explicit MPPI(const mjModel* m,
                  const QRCost<state_size, ctrl_size>& cost,
                  const MPPIParams& params);

    ~MPPI();

    void control(const mjData* d);

    ctrl_vector _cached_control;

private:

    MPPI<state_size, ctrl_size>::ctrl_vector  total_entropy(const std::vector<ctrl_vector>& delta_control_samples,
                                                            const std::vector<double>& d_cost_to_go_samples) const;

    void compute_control_trajectory();

    const MPPIParams& m_params;
    QRCost<state_size, ctrl_size> m_cost_func;
    std::vector<state_vector> m_state;

    std::vector<ctrl_vector> m_control;
    std::vector<double> m_delta_cost_to_go;

    std::vector<std::vector<ctrl_vector>> m_delta_control;
    const mjModel* m_m;

    mjData*  m_d_cp = nullptr;
    [[maybe_unused]] std::vector<mjtNum> m_cost;
};


#endif //OPTCONTROL_MUJOCO_MPPI_H
