#ifndef OPTCONTROL_MUJOCO_ILQR_H
#define OPTCONTROL_MUJOCO_ILQR_H

#include <vector>
#include "mjdata.h"
#include "Eigen/Core"
#include "../utilities/internal_types.h"
#include "../parameters/simulation_params.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"


struct ILQRParams
{
    double m_min_reg = 1e-6;
    double _prev_total_cost  = 0;
    double _delta_init = 2.0;
    double m_min_cost_red = 5;
};


template<int state_size, int ctrl_size>
class ILQR
{
    using ctrl_vec  = Eigen::Matrix<double, ctrl_size, 1>;
    using state_vec = Eigen::Matrix<double, state_size, 1>;
    using state_mat = Eigen::Matrix<double, state_size, state_size>;
    using ctrl_mat  = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_ctrl_mat = Eigen::Matrix<double, state_size, ctrl_size>;
    using ctrl_state_mat = Eigen::Matrix<double, ctrl_size, state_size>;

public:

    ILQR(FiniteDifference<state_size, ctrl_size>& fd,
         CostFunction<state_size, ctrl_size>& cf,
         const mjModel * m,
         int simulation_time,
         int iteration,
         const mjData* d,
         const std::vector<ctrl_vec>* init_u = nullptr);

    ~ILQR();
    void control(const mjData* d);

private:

    void forward_simulate(const mjData* d);
    void forward_pass(const mjData* d);
    void update_regularizer(const bool increase);
    double compute_expected_cost(const double backtracker);
    void backward_pass();
    ctrl_vec       Q_u(int time, Eigen::Matrix<double, state_size, 1>& _v_x);
    state_vec      Q_x(int time, Eigen::Matrix<double, state_size, 1>& _v_x);
    state_mat      Q_xx(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    ctrl_state_mat Q_ux(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    state_ctrl_mat Q_xu(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    ctrl_mat       Q_uu(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    ctrl_state_mat Q_ux_reg(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    state_ctrl_mat Q_xu_reg(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    ctrl_mat       Q_uu_reg(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);


    std::array<double, 10> _backtrackers{};
    state_mat _regularizer;
    const double m_regulariser_min = 1e-6;

    // Control containers
    std::vector<state_vec>      _x_traj;
    std::vector<state_vec>      _x_traj_new;
    std::vector<ctrl_vec>       _u_traj_new;

    struct Derivatives{
        double l; StateVector lx; StateMatrix lxx; CtrlVector lu; CtrlMatrix luu;
        CtrlStateMatrix lux;StateMatrix fx; StateCtrlMatrix fu;
    };

    struct BackPassVars
    {
        CtrlVector ff_k; CtrlStateMatrix fb_k;
    };

    std::vector<Derivatives> m_d_vector;
    std::vector<BackPassVars> m_bp_vector;

    //HJB Approximation
    std::vector<CtrlVector> m_Qu_traj;
    std::vector<CtrlMatrix> m_Quu_traj;

    FiniteDifference<state_size, ctrl_size>& _fd;
    CostFunction<state_size, ctrl_size>&     _cf;

    const mjModel* _m;
    mjData* _d_cp            = nullptr;
    double _prev_total_cost  = 0;
    const double _delta_init = 1.6;
    const double m_min_cost_red = 0;
    double _delta            = _delta_init;

public:
    ctrl_vec              _cached_control;
    std::vector<ctrl_mat> _covariance;
    std::vector<ctrl_vec> _u_traj;
    std::vector<ctrl_vec> _u_traj_cp;
    std::vector<double>   cost;
    std::vector<double>   exp_cost_reduction;
    const int             _simulation_time;
    const int             _iteration;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
