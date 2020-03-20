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
    void backward_pass(mjData* d);
    void forward_pass();
    Eigen::Ref<InternalTypes::Mat2x1> get_control();
    std::vector<InternalTypes::Mat2x1> _u_traj;

private:
    void forward_simulate(const mjData* d);
    Eigen::Matrix<mjtNum, 4, 1> Q_x(int time, InternalTypes::Mat4x1& _V_x);
    Eigen::Matrix<mjtNum, 2, 1> Q_u(int time, InternalTypes::Mat4x1& _V_x);
    Eigen::Matrix<mjtNum, 4, 4> Q_xx(int time, InternalTypes::Mat4x4& _V_xx);
    Eigen::Matrix<mjtNum, 2, 4> Q_ux(int time, InternalTypes::Mat4x4& _V_xx);
    Eigen::Matrix<mjtNum, 2, 2> Q_uu(int time, InternalTypes::Mat4x4& _V_xx);

    std::vector<double> _F;
    std::vector<InternalTypes::Mat4x4> _F_x;
    std::vector<InternalTypes::Mat4x2> _F_u;

    std::vector<mjtNum> _L;
    std::vector<InternalTypes::Mat4x1> _L_x;
    std::vector<InternalTypes::Mat2x1> _L_u;
    std::vector<InternalTypes::Mat4x4> _L_xx;
    std::vector<InternalTypes::Mat2x4> _L_ux;
    std::vector<InternalTypes::Mat2x2> _L_uu;

//    std::vector<InternalTypes::Mat2x1> _u_traj;
    std::vector<InternalTypes::Mat2x4> _ff_K;
    std::vector<InternalTypes::Mat4x1> _x_traj;
    std::vector<InternalTypes::Mat2x1> _fb_k ;
    std::vector<InternalTypes::Mat4x1> _x_traj_new;

    InternalTypes::Mat6x1 desired_state;
    int  _simulation_time;

    const mjModel* _m;
    mjData* _d_cp = nullptr;

    CostFunction& _cf;
    FiniteDifference& _fd;
    InternalTypes::Mat2x1 _cached_control;

    mjtNum min_bound = -3;
    mjtNum max_bound = 3;

    bool recalculate = true;
    bool converged   = false;
};


#endif //OPTCONTROL_MUJOCO_ILQR_H
