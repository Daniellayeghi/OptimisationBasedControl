
#ifndef DRAKE_CMAKE_INSTALLED_CONTROLLER_H
#define DRAKE_CMAKE_INSTALLED_CONTROLLER_H

#include "mujoco.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"
#include "ilqr.h"
#include "MPPI.h"


template<typename T, int state_size, int ctrl_size>
class MyController
{
public:
    MyController(const mjModel *m, mjData *d, const T& controls);

    void controller();

    static void set_instance(MyController *my_ctrl);

    static void callback_wrapper(const mjModel* m, mjData* d);

    static void dummy_controller(const mjModel* m, mjData* d);

    std::vector<Eigen::Matrix<double, ctrl_size, 1>> ctrl_buffer;

private:
    const T& controls;
    const mjModel* _m;
    mjData* _d;
};

#endif //DRAKE_CMAKE_INSTALLED_CONTROLLER_H
