
#include <iostream>
#include <chrono>
#include <random>
#include "controller.h"
#include "../utilities/finite_diff.h"
#include "../utilities/internal_types.h"
#include "cost_function.h"
#include "simulation_params.h"

using namespace SimulationParameters;
static MyController<n_jvel + n_jpos, n_ctrl> *my_ctrl;


template<int state_size, int ctrl_size>
MyController<state_size, ctrl_size>::MyController(const mjModel *m, mjData *d, ILQR& ilqr, const MPPI<state_size, ctrl_size>& pi) :
_pi(pi), _ilqr(ilqr), _m(m), _d(d)
{
    _inertial_torque = mj_stackAlloc(_d, _m->nv);
    _constant_acc = mj_stackAlloc(d, m->nv);

    ctrl_buffer.assign(10, Eigen::Matrix<double, ctrl_size, 1>::Zero());

    for (std::size_t row = 0; row < 3; ++row)
    {
        _constant_acc[row] = 0;
    }
}


template<int state_size, int ctrl_size>
void MyController<state_size, ctrl_size>::controller()
{
#if 0
    mj_mulM(_m, _d, _inertial_torque, _constant_acc);
    _d->ctrl[0] = _d->qfrc_bias[0] + _inertial_torque[0];
    _d->ctrl[1] = _d->qfrc_bias[1] + _inertial_torque[1];
    _d->ctrl[2] = _d->qfrc_bias[2] + _inertial_torque[2];
#endif

#if 0
    auto ctrl = _ilqr.get_control();
    std::cout << "Ctrl: " << "\n" << ctrl << "\n";
    _d->ctrl[1] = ctrl;
#endif

#if 1
//    std::cout << "Control: " << _pi._cached_control << std::endl;
    for (auto row = 0; row < ctrl_size; ++row)
    {
        _d->ctrl[row] = _pi._cached_control(row, 0);
    }

    ctrl_buffer.emplace_back(_pi._cached_control);
    ++iteration;
    if (iteration % 25 == 0)
    {
        std::cout << iteration <<"\n";
    }

#endif


#ifdef DEFINE_DEBUG
    std::cout << "Vel__0: " << _d->qvel[0] << "\n";
    std::cout << "Vel__1: " << _d->qvel[1] << "\n";
#endif
}


template<int state_size, int ctrl_size>
void MyController<state_size, ctrl_size>::set_instance(MyController<state_size, ctrl_size> *myctrl)
{
    my_ctrl = myctrl;
}


template<int state_size, int ctrl_size>
void MyController<state_size,ctrl_size>::callback_wrapper(const mjModel *m, mjData *d)
{
    my_ctrl->controller();
}


template<int state_size, int ctrl_size>
void MyController<state_size, ctrl_size>::dummy_controller(const mjModel *m, mjData *d)
{

}


template class MyController<n_jpos + n_jvel, n_ctrl>;
