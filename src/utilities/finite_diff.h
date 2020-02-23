
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

    void differentiate(mjData *d, mjtNum *wrt, const WithRespectTo id);
    mjtNum* get_wrt(const WithRespectTo wrt);

private:
    void copy_state(const mjData* d);
    void first_order_forward_diff_general(mjtNum* target, const mjtNum* original, const mjtNum* output,
                                          const mjtNum* center, Eigen::Matrix<mjtNum, 3, 3>& result);
    void first_order_forward_diff_positional(mjtNum* target, const mjtNum* original, const mjtNum* output,
                                             const mjtNum* center, Eigen::Matrix<mjtNum, 3, 3>& result);

    const mjModel* _m = nullptr;
    mjData* _d_cp     = nullptr;
    mjtNum* _f_du     = nullptr;
    const double eps  = 1e-6;

    std::map<WithRespectTo, mjtNum *> _wrt;
    std::map<WithRespectTo, mjtStage> _skip;
};

#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H
