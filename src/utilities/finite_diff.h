

#ifndef OPTCONTROL_MUJOCO_FINITE_DIFF_H
#define OPTCONTROL_MUJOCO_FINITE_DIFF_H

#include <map>
#include <vector>
#include <mujoco/mujoco.h>
#include "string"
#include "eigen3/Eigen/Dense"
#include "../parameters/simulation_params.h"
#include "../utilities/generic_utils.h"

using namespace GenericUtils;
using namespace SimulationParameters;
class FiniteDifference
{
public:
    using ctrl_jacobian          = StateCtrlMatrix;
    using state_vel_jacobian     = Eigen::Matrix<double, state_size, n_jvel>;
    using state_pos_jacobian     = Eigen::Matrix<double, state_size, n_jpos>;
    using complete_jacobian      = Eigen::Matrix<double, state_size, state_size + n_ctrl>;

    enum class WithRespectTo{ACC, VEL, POS, CTRL, FRC};
    explicit FiniteDifference(const mjModel* m);
    ~FiniteDifference();
    Eigen::Block<complete_jacobian, state_size, state_size> f_x();
    Eigen::Block<complete_jacobian, state_size, n_ctrl> f_u();
    void f_x_f_u(mjData *d);

private:
    struct FDFuncArgs{mjtNum* target; const mjtNum* original; const mjtNum* centre_pos; const mjtNum* centre_vel;};
    FastPair<mjtNum *, mjtNum *> set_finite_diff_arguments(const mjData *d,mjtNum *wrt,WithRespectTo id,bool do_copy = true);
    void diff_wrt(const mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy = true);
    void perturb_target(mjtNum *target, WithRespectTo id, int state_iter);
    ctrl_jacobian finite_diff_wrt_ctrl(const FDFuncArgs& fd_args, const mjData *d, WithRespectTo id);
    state_vel_jacobian finite_diff_wrt_state_vel(const FDFuncArgs& fd_args, const mjData *d, WithRespectTo id);
    state_pos_jacobian finite_diff_wrt_state_pos(const FDFuncArgs& fd_args,const mjData *d, WithRespectTo id);

    // Internal Jacobian Blocks
    state_pos_jacobian sp_jac = state_pos_jacobian::Zero();
    state_vel_jacobian sv_jac = state_vel_jacobian::Zero();
    ctrl_jacobian ctrl_jac = ctrl_jacobian::Zero();
    complete_jacobian _full_jacobian = complete_jacobian::Zero();

    std::map<WithRespectTo, mjtNum *> _wrt;

    const mjModel* _m = nullptr;
    mjData* _d_cp     = nullptr;
    const double eps  = .000001;
};

#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H