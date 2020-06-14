
#ifndef DRAKE_CMAKE_INSTALLED_CONTROLLER_H
#define DRAKE_CMAKE_INSTALLED_CONTROLLER_H

#include "mujoco.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"
#include "ilqr.h"
#include "MPPI.h"


template<int state_size, int ctrl_size>
class MyController
{
public:
    MyController(const mjModel *m, mjData *d, ILQR<state_size, ctrl_size>& ilqr, const MPPI<state_size, ctrl_size>& pi);

    void controller();

    static void set_instance(MyController *my_ctrl);

    static void callback_wrapper(const mjModel* m, mjData* d);

    static void dummy_controller(const mjModel* m, mjData* d);

    std::vector<Eigen::Matrix<double, ctrl_size, 1>> ctrl_buffer;

private:
    const MPPI<state_size, ctrl_size>& _pi;
    ILQR<state_size, ctrl_size>&             _ilqr;
    const mjModel*    _m;
    mjData* _d;
    mjtNum* _inertial_torque;
    mjtNum* _constant_acc;
    int iteration = 0;
};

#endif //DRAKE_CMAKE_INSTALLED_CONTROLLER_H
