
#include <iostream>
#include <chrono>
#include <random>
#include "controller.h"
#include "../utilities/finite_diff.h"
#include "../utilities/internal_types.h"
#include "cost_function.h"

static MyController *my_ctrl;
static std::default_random_engine generator;
static std::uniform_real_distribution<double> distribution(-.2,.2);


MyController::MyController(const mjModel *m, mjData *d, FiniteDifference& fd, CostFunction& cf) :
_m(m), _d(d), _fd(fd), _cf(cf), _ilqr(_fd, _cf, _m, 100)
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
    _d->ctrl[0] = _d->qfrc_bias[0] + _inertial_torque[0] + distribution(generator);
    _d->ctrl[1] = _d->qfrc_bias[1] + _inertial_torque[1] + distribution(generator);
    _d->ctrl[2] = _d->qfrc_bias[2] + _inertial_torque[2] + distribution(generator);
    _d->ctrl[3] = _d->qfrc_bias[3] + _inertial_torque[3] + distribution(generator);


//    std::cout << "Fux" << "\n" << result << std::endl;
//    std::cout << "Lx: " << "\n" << _cf.L_x() << "\n";
//    std::cout << "Lu: " << "\n" << _cf.L_u() << "\n";
//    std::cout << "Lux: " << "\n" << _cf.L_ux() << "\n";
//    std::cout << "Lxx: " << "\n" << _cf.L_xx() << "\n";
//    std::cout << "Luu: " << "\n" << _cf.L_uu() << "\n";

#ifdef DEFINE_DEBUG
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
