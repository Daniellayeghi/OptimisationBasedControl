#ifndef OPTCONTROL_MUJOCO_ILQR_H
#define OPTCONTROL_MUJOCO_ILQR_H

#include <vector>
#include "mjdata.h"
#include "Eigen/Core"
#include "../utilities/internal_types.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"


class ILQR
{
public:
    ILQR(FiniteDifference& fd, CostFunction& cf, const mjModel * m, int simulation_time);
    ~ILQR();

    void control(mjData* d);
    void backward_pass();
    void forward_pass();
    double get_control();
    std::vector<double> _u_traj;

private:
    void forward_simulate(const mjData* d);
    double Q_u(int time, InternalTypes::Mat4x1& _v_x);
    double Q_uu(int time, InternalTypes::Mat4x4& _v_xx);
    Eigen::Matrix<mjtNum, 4, 1> Q_x(int time, InternalTypes::Mat4x1& _v_x);
    Eigen::Matrix<mjtNum, 4, 4> Q_xx(int time, InternalTypes::Mat4x4& _v_xx);
    Eigen::Matrix<mjtNum, 1, 4> Q_ux(int time, InternalTypes::Mat4x4& _v_xx);

    std::vector<double> _f;
    std::vector<InternalTypes::Mat4x4> _f_x;

    std::vector<InternalTypes::Mat4x2> _f_u;
    std::vector<mjtNum> _l;
    std::vector<InternalTypes::Mat4x1> _l_x;
    std::vector<InternalTypes::Mat2x1> _l_u;
    std::vector<InternalTypes::Mat4x4> _l_xx;
    std::vector<InternalTypes::Mat2x4> _l_ux;
    std::vector<InternalTypes::Mat2x2> _l_uu;
    std::vector<Eigen::Matrix<mjtNum, 1, 4>> _fb_K;
    std::vector<InternalTypes::Mat4x1> _x_traj;
    std::vector<double> _ff_k ;

    std::vector<InternalTypes::Mat4x1> _x_traj_new;
    InternalTypes::Mat6x1 desired_state;

    int  _simulation_time;
    const mjModel* _m;

    mjData* _d_cp = nullptr;
    CostFunction& _cf;
    FiniteDifference& _fd;

    double _cached_control;
    bool recalculate = true;
    bool converged   = false;

    mjtNum min_bound = -1;
    mjtNum max_bound = 1;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
