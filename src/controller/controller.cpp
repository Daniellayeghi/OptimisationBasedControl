
#include <iostream>
#include <chrono>
#include "controller.h"

static MyController *my_ctrl;

#include <iostream>
#include "autodiff/forward.hpp"

using namespace std;
using namespace autodiff;

// The multi-variable function for which derivatives are needed
dual f(dual x, dual y, dual z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}


MyController::MyController(const mjModel *m, mjData *d) : _m(m), _d(d)
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
    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    dual u = f(x, y, z);

    double dudx = derivative(f, wrt(x), at(x, y, z));
    double dudy = derivative(f, wrt(y), at(x, y, z));
    double dudz = derivative(f, wrt(z), at(x, y, z));

    cout << "u = " << u << endl;
    cout << "du/dx = " << dudx << endl;
    cout << "du/dy = " << dudy << endl;
    cout << "du/dz = " << dudz << endl;

    mj_mulM(_m, _d, _inertial_torque, _constant_acc);

    _d->qfrc_applied[0] = _d->qfrc_bias[0] + _inertial_torque[0];
    _d->qfrc_applied[1] = _d->qfrc_bias[1] + _inertial_torque[0];
    _d->qfrc_applied[2] = _d->qfrc_bias[2] + _inertial_torque[0];
}


void MyController::set_instance(MyController *myctrl)
{
    my_ctrl = myctrl;
}


void MyController::callback_wrapper(const mjModel *m, mjData *d)
{
    my_ctrl->controller();
}