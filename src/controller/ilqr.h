#ifndef OPTCONTROL_MUJOCO_ILQR_H
#define OPTCONTROL_MUJOCO_ILQR_H

#include <vector>
#include "mjdata.h"
#include "Eigen/Core"
#include "../utilities/internal_types.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"

template<int state_size, int ctrl_size>
class ILQR
{
    using ilqr_t    = ILQR<state_size, ctrl_size>;
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
    void backward_pass();
    void forward_pass(const mjData* d);

private:
    void forward_simulate(const mjData* d);
    ctrl_vec       Q_u(int time, Eigen::Matrix<double, state_size, 1>& _v_x);
    ctrl_mat       Q_uu(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    state_vec      Q_x(int time, Eigen::Matrix<double, state_size, 1>& _v_x);
    state_mat      Q_xx(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    ctrl_state_mat Q_ux(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);

    std::array<double, 10> _backtrackers{};
    state_mat _regularizer;

    std::vector<state_mat> _f_x;
    std::vector<state_ctrl_mat> _f_u;
    std::vector<double> _l;
    std::vector<state_vec> _l_x;
    std::vector<ctrl_vec> _l_u;
    std::vector<state_mat> _l_xx;

    std::vector<ctrl_state_mat> _l_ux;
    std::vector<ctrl_mat> _l_uu;

    std::vector<ctrl_vec> _ff_k ;
    std::vector<ctrl_state_mat> _fb_K;
    std::vector<state_vec> _x_traj;

    std::vector<state_vec> _x_traj_new;
    std::vector<ctrl_vec>  _u_traj_new;

    FiniteDifference<state_size, ctrl_size>& _fd;
    CostFunction<state_size, ctrl_size>& _cf;

    const mjModel* _m;
    mjData* _d_cp            = nullptr;
    double _prev_total_cost  = 0;
    const double _delta_init = 2.0;
    double _delta            = _delta_init;

    bool converged   = false;
    bool accepted    = false;
    bool recalculate = true;

    mjtNum min_bound = -1;
    mjtNum max_bound = 1;

public:

    ctrl_vec _cached_control;
    std::vector<ctrl_vec> _u_traj;
    std::vector<double>   cost;
    std::vector<double>   exp_cost_reduction;
    const int _simulation_time;
    const int _iteration;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
