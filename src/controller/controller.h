
#ifndef DRAKE_CMAKE_INSTALLED_CONTROLLER_H
#define DRAKE_CMAKE_INSTALLED_CONTROLLER_H

#include "mujoco.h"
#include "../parameters/simulation_params.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"
#include "ilqr.h"


template<typename T, int state_size, int ctrl_size>
class MyController
{
public:
    MyController(const mjModel *m, mjData *d, const T& controls, const bool comp_gravity = false);

    void controller();

    static void set_instance(MyController *my_ctrl);

    static void callback_wrapper(const mjModel* m, mjData* d);

    static void dummy_controller(const mjModel* m, mjData* d);

private:
    int iteration = 0;
    const T& controls;
    const mjModel* _m;
    mjData* _d;
    bool m_comp_gravity;
    Eigen::Map<CtrlVector> m_grav_comp;
};

#endif //DRAKE_CMAKE_INSTALLED_CONTROLLER_H
