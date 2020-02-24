
#ifndef OPTCONTROL_MUJOCO_FINITE_DIFF_H
#define OPTCONTROL_MUJOCO_FINITE_DIFF_H

#include <map>
#include <vector>
#include "mujoco.h"
#include "string"
#include "eigen3/Eigen/Dense"

class FiniteDifference
{
public:
    enum class WithRespectTo{ACC, VEL, POS, CTRL};

    explicit FiniteDifference(const mjModel* m);

    ~FiniteDifference();

    Eigen::Matrix<mjtNum, 9, 1> differentiate(mjData *d, mjtNum *wrt, const WithRespectTo id);

    Eigen::Matrix<mjtNum, 9, 1> f_u(mjData *d);

    Eigen::Matrix<mjtNum, 9, 2> f_x(mjData *d);

    Eigen::Matrix<mjtNum, 9, 3> f_x_f_u(mjData *d);

    mjtNum* get_wrt(const WithRespectTo wrt);

private:
    void copy_state(const mjData* d);

    Eigen::Matrix<mjtNum, 9, 1> first_order_forward_diff_general(mjtNum* target,
                                                                 const mjtNum* original,
                                                                 const mjtNum* output,
                                                                 const mjtNum* center,
                                                                 const mjtStage skip);

    Eigen::Matrix<mjtNum, 9, 1> first_order_forward_diff_positional(mjtNum* target,
                                                                    const mjtNum* original,
                                                                    const mjtNum* output,
                                                                    const mjtNum* center,
                                                                    const mjtStage skip);

    const mjModel* _m = nullptr;
    mjData* _d_cp     = nullptr;
    mjtNum* _f_du     = nullptr;
    const double eps  = 1e-6;

    std::map<WithRespectTo, mjtNum *> _wrt;
};

#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H
