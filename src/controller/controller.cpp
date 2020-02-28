
#include <iostream>
#include <chrono>
#include "controller.h"
#include "../utilities/finite_diff.h"
#include "cost_function.h"

static MyController *my_ctrl;


MyController::MyController(const mjModel *m, mjData *d, FiniteDifference& fd, CostFunction& cf) :
_m(m), _d(d), _fd(fd), _cf(cf)
{
    _inertial_torque = mj_stackAlloc(_d, _m->nv);
    _constant_acc = mj_stackAlloc(d, m->nv);
    for (std::size_t row = 0; row < 3; ++row)
    {
        _constant_acc[row] = 0.4;
    }
}


void MyController::controller()
{
    mj_mulM(_m, _d, _inertial_torque, _constant_acc);
    _d->ctrl[0] = _d->qfrc_bias[0] + _inertial_torque[0];
    _d->ctrl[1] = _d->qfrc_bias[1] + _inertial_torque[1];
    _d->ctrl[2] = _d->qfrc_bias[2] + _inertial_torque[2];
}


void MyController::set_instance(MyController *myctrl)
{
    my_ctrl = myctrl;
}


void MyController::callback_wrapper(const mjModel *m, mjData *d)
{
    my_ctrl->controller();
}


void MyController::dummy_controller(const mjModel *m, mjData *d)
{

}