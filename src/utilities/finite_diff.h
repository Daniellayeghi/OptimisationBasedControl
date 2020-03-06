
#ifndef OPTCONTROL_MUJOCO_FINITE_DIFF_H
#define OPTCONTROL_MUJOCO_FINITE_DIFF_H

#include <map>
#include <vector>
#include "mujoco.h"
#include "string"
#include "eigen3/Eigen/Dense"
#include "internal_types.h"

class FiniteDifference
{
public:
    enum class WithRespectTo{ACC, VEL, POS, CTRL, FRC};

    explicit FiniteDifference(const mjModel* m);

    ~FiniteDifference();

    InternalTypes::Mat3x3 differentiate(mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy = true);

    Eigen::Matrix<mjtNum, 3, 3> f_u(mjData *d);

    Eigen::Matrix<mjtNum, 3, 6> f_x(mjData *d);

    void f_x_f_u(mjData *d);

    mjtNum* get_wrt(WithRespectTo wrt);

    Eigen::Matrix<mjtNum, 3, 9>& get_full_derivatives();


private:
    void copy_state(const mjData* d);

    InternalTypes::Mat3x3 first_order_forward_diff_general(mjtNum* target,
                                                           const mjtNum* original,
                                                           const mjtNum* output,
                                                           const mjtNum* center,
                                                           mjtStage skip);

    InternalTypes::Mat3x3 first_order_forward_diff_positional(mjtNum* target,
                                                             const mjtNum* original,
                                                             const mjtNum* output,
                                                             const mjtNum* center,
                                                             mjtStage skip);

    const mjModel* _m = nullptr;
    mjData* _d_cp     = nullptr;
    const double eps  = 1e-6;

    std::map<WithRespectTo, mjtNum *> _wrt;
    InternalTypes::Mat3x9 _full_jacobian;
};

#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H
