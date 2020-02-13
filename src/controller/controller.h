
#ifndef DRAKE_CMAKE_INSTALLED_CONTROLLER_H
#define DRAKE_CMAKE_INSTALLED_CONTROLLER_H

#include "mujoco.h"

class MyController
{
public:
    MyController(const mjModel *m, mjData *d);

    void controller();

    static void set_instance(MyController *my_ctrl);

    static void callback_wrapper(const mjModel* m, mjData* d);


private:
    const mjModel* _m;
    mjData* _d;
    mjtNum* _inertial_torque;
    mjtNum* _constant_acc;
};

#endif //DRAKE_CMAKE_INSTALLED_CONTROLLER_H
