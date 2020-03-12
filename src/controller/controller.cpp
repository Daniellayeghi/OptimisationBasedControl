
#include <iostream>
#include <chrono>
#include <random>
#include "controller.h"
#include "../utilities/finite_diff.h"
#include "../utilities/internal_types.h"
#include "cost_function.h"

static MyController *my_ctrl;
static std::default_random_engine generator;
static std::uniform_real_distribution<double> distribution(-.01,.01);


MyController::MyController(const mjModel *m, mjData *d, FiniteDifference& fd, CostFunction& cf, ILQR& ilqr) :
_m(m), _d(d), _fd(fd), _cf(cf), _ilqr(ilqr)
{
    _inertial_torque = mj_stackAlloc(_d, _m->nv);
    _constant_acc = mj_stackAlloc(d, m->nv);
    for (std::size_t row = 0; row < 3; ++row)
    {
        _constant_acc[row] = 0;
    }
}


void MyController::controller()
{
    mj_mulM(_m, _d, _inertial_torque, _constant_acc);
#if 0
    _d->ctrl[0] = _d->qfrc_bias[0] + _inertial_torque[0];
    _d->ctrl[1] = _d->qfrc_bias[1] + _inertial_torque[1];
    _d->ctrl[2] = _d->qfrc_bias[2] + _inertial_torque[2];
#endif
    auto ctrl = _ilqr.get_control();
    std::cout << "Ctrl: " << "\n" << ctrl << "\n";
    _d->ctrl[0] = 0;
    _d->ctrl[1] = 0;

#ifdef DEFINE_DEBUG
    for (auto joint = 0; joint < 3; ++joint)
    {
        std::cout << "Pos: " << joint << " " << _d->qpos[joint] << "\n";
        std::cout << "Vel: " << joint << " " << _d->qvel[joint] << "\n";
    }
    std::cout << "LQR dynamics derivatives: " << result << "\n";
#endif
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
