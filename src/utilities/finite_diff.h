
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

    InternalTypes::Mat4x2 differentiate(mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy = true);

    InternalTypes::Mat4x2 f_u(mjData *d);

    InternalTypes::Mat4x4 f_x(mjData *d);

    void f_x_f_u(mjData *d);

    mjtNum* get_wrt(WithRespectTo wrt);

    InternalTypes::Mat4x6& get_full_derivatives();


private:
    void copy_state(const mjData* d);

    InternalTypes::Mat4x2 first_order_forward_diff_general(mjtNum* target,
                                                           const mjtNum* original,
                                                           const mjtNum* output,
                                                           const mjtNum* center_pos,
                                                           const mjtNum* center_vel,
                                                           mjtStage skip);

    InternalTypes::Mat4x2 first_order_forward_diff_positional(mjtNum* target,
                                                             const mjtNum* original,
                                                             const mjtNum* output,
                                                             const mjtNum* center_pos,
                                                             const mjtNum* center_vel,
                                                             mjtStage skip);

    const mjModel* _m = nullptr;
    mjData* _d_cp     = nullptr;
    const double eps  = 1e-2;

    std::map<WithRespectTo, mjtNum *> _wrt;
    InternalTypes::Mat4x6 _full_jacobian;
};

#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H
