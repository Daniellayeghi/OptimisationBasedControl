
#ifndef DRAKE_CMAKE_INSTALLED_CONTROLLER_H
#define DRAKE_CMAKE_INSTALLED_CONTROLLER_H

#include "mujoco.h"

class MyController
{
public:
    MyController(const mjModel *m, const mjModel *m_cp, mjData *d, mjData *d_cp);

    ~MyController();

    void controller();

    static void set_instance(MyController *my_ctrl);

    static void callback_wrapper(const mjModel* m, mjData* d);

private:
    const mjModel* _m;
    const mjModel* _m_cp;
    mjData* _d;
    mjData* _d_cp;
    mjtNum* _inertial_torque;
    mjtNum* _constant_acc;
    mjtNum* f_duu = nullptr;
    mjtNum* f_du  = nullptr;
};

#endif //DRAKE_CMAKE_INSTALLED_CONTROLLER_H
