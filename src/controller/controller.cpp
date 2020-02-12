
#include <iostream>
#include "controller.h"

MyController::MyController(const mjModel *m, mjData *d) : _m(m), _d(d)
{
    _inertial_torque = mj_stackAlloc(_d, _m->nv);
    _constant_acc = mj_stackAlloc(d, m->nv);
    for (std::size_t row = 0; row < 3; ++row)
    {
        _constant_acc[row] = 1;
    }
}


void MyController::controller()
{
    mj_sensorPos(_m, _d);

    mj_mulM(_m, _d, _inertial_torque, _constant_acc);

    std::cout << _d->qfrc_bias[0] << " " << _d->qfrc_bias[1] << " " <<_d->qfrc_bias[2] << std::endl;

    _d->qfrc_applied[0] = _d->qfrc_bias[0] + _inertial_torque[0];
    _d->qfrc_applied[1] = _d->qfrc_bias[1] + _inertial_torque[0];
    _d->qfrc_applied[2] = _d->qfrc_bias[2] + _inertial_torque[0];
}