
#ifndef OPTCONTROL_MUJOCO_FINITE_DIFF_H
#define OPTCONTROL_MUJOCO_FINITE_DIFF_H

#include <map>
#include <vector>
#include "mujoco.h"
#include "string"
#include "eigen3/Eigen/Dense"
#include "internal_types.h"


template<int state_size, int ctrl_size>
class FiniteDifference
{
    using ctrl_jacobian          = Eigen::Matrix<double, state_size/2, ctrl_size>;
    using full_state_jacobian    = Eigen::Matrix<double, state_size/2, state_size>;
    using partial_state_jacobian = Eigen::Matrix<double, state_size/2, state_size/2>;
    using complete_jacobian      = Eigen::Matrix<double, state_size/2, state_size + ctrl_size>;
public:

    enum class WithRespectTo{ACC, VEL, POS, CTRL, FRC};

    explicit FiniteDifference(const mjModel* m);

    ~FiniteDifference();

    Eigen::Block<complete_jacobian, state_size/2, state_size> f_x();

    Eigen::Block<complete_jacobian, state_size/2, ctrl_size> f_u();

    void f_x_f_u(mjData *d);

private:

    ctrl_jacobian diff_wrt_ctrl(mjData *d,
                                mjtNum *wrt,
                                WithRespectTo id,
                                bool do_copy = true);


    partial_state_jacobian diff_wrt_state(mjData *d,
                                          mjtNum *wrt,
                                          WithRespectTo id,
                                          bool do_copy = true);


    ctrl_jacobian finite_diff_wrt_ctrl(mjtNum* target,
                                       const mjtNum* original,
                                       const mjtNum* centre_acc,
                                       const mjData *d,
                                       const WithRespectTo id);


    partial_state_jacobian finite_diff_wrt_state(mjtNum* target,
                                                 const mjtNum* original,
                                                 const mjtNum* centre_acc,
                                                 const mjData *d,
                                                 const WithRespectTo id);

    const mjModel* _m = nullptr;

    mjData* _d_cp     = nullptr;
    const double eps  = 0.000001;
    std::map<WithRespectTo, mjtNum *> _wrt;

    complete_jacobian _full_jacobian;
};

#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H
