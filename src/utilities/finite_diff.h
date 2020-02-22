
#ifndef OPTCONTROL_MUJOCO_FINITE_DIFF_H
#define OPTCONTROL_MUJOCO_FINITE_DIFF_H

#include "mujoco.h"
#include "string"

class FiniteDifference
{
public:
    explicit FiniteDifference(const mjModel* m);
    ~FiniteDifference();
    void f_u(const mjData *d);

private:
    void copy_state(const mjData* d);
    void first_order_forward_diff(mjtNum* target, const mjtNum* original, const mjtNum* output, const mjtNum* center);

    const mjModel* _m = nullptr;
    mjData* _d_cp     = nullptr;
    mjtNum* _f_du     = nullptr;
    const double eps  = 1e-6;
};


#endif //OPTCONTROL_MUJOCO_FINITE_DIFF_H
