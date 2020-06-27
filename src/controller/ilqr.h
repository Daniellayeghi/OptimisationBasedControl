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
public:

    ILQR(FiniteDifference<state_size, ctrl_size>& fd, CostFunction<state_size, ctrl_size>& cf, const mjModel * m, const int simulation_time);

    ~ILQR();

    void control(mjData* d);
    void backward_pass();
    void forward_pass();
    Eigen::Matrix<double, ctrl_size, 1> _cached_control;
    std::vector<Eigen::Matrix<double, ctrl_size, 1>> _u_traj;
private:

    void forward_simulate(const mjData* d);
    Eigen::Matrix<double, ctrl_size, 1> Q_u(int time, Eigen::Matrix<double, state_size, 1>& _v_x);


    Eigen::Matrix<double, ctrl_size, ctrl_size> Q_uu(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    Eigen::Matrix<double, state_size, 1> Q_x(int time, Eigen::Matrix<double, state_size, 1>& _v_x);
    Eigen::Matrix<double, state_size, state_size> Q_xx(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);

    Eigen::Matrix<double, ctrl_size, state_size> Q_ux(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx);
    Eigen::Matrix<double, state_size, state_size> _regularizer;

    std::array<double, 10> _backtrackers{};
    std::vector<Eigen::Matrix<double, state_size, state_size>> _f_x;

    std::vector<Eigen::Matrix<double, state_size, ctrl_size>> _f_u;
    std::vector<double> _l;
    std::vector<Eigen::Matrix<double, state_size, 1>> _l_x;
    std::vector<Eigen::Matrix<double, ctrl_size, 1>> _l_u;
    std::vector<Eigen::Matrix<double, state_size, state_size>> _l_xx;
    std::vector<Eigen::Matrix<double, ctrl_size, state_size>> _l_ux;

    std::vector<Eigen::Matrix<double, ctrl_size, ctrl_size>> _l_uu;
    std::vector<Eigen::Matrix<double, ctrl_size, 1>> _ff_k ;

    std::vector<Eigen::Matrix<mjtNum, ctrl_size, state_size>> _fb_K;
    std::vector<Eigen::Matrix<double, state_size, 1>> _x_traj;

    std::vector<Eigen::Matrix<double, state_size, 1>> _x_traj_new;
    double _prev_total_cost;
    int    _simulation_time;
    const double _delta_init = 2.0;
    double _delta = _delta_init;

    const mjModel*    _m;
    mjData* _d_cp = nullptr;
    CostFunction<state_size, ctrl_size>& _cf;
    FiniteDifference<state_size, ctrl_size>& _fd;

    bool recalculate = true;
    bool converged   = false;
    bool accepted    = false;

    mjtNum min_bound = -1;
    mjtNum max_bound = 1;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
