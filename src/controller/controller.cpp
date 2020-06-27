
#include <iostream>
#include <chrono>
#include <random>
#include "controller.h"
#include "../utilities/finite_diff.h"
#include "../utilities/internal_types.h"
#include "cost_function.h"
#include "simulation_params.h"

using namespace SimulationParameters;
static MyController<MPPI<n_jvel + n_jpos, n_ctrl>, n_jvel + n_jpos, n_ctrl> *my_ctrl_mppi;
static MyController<ILQR<n_jvel + n_jpos, n_ctrl>, n_jvel + n_jpos, n_ctrl> *my_ctrl_ilqr ;


template<typename T, int state_size, int ctrl_size>
MyController<T, state_size, ctrl_size>::MyController(const mjModel *m, mjData *d, const T& controls) :
controls(controls), _m(m), _d(d)
{
    ctrl_buffer.assign(10, Eigen::Matrix<double, ctrl_size, 1>::Zero());
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::controller()
{
#if 1
    for (auto row = 0; row < ctrl_size; ++row)
    {
        _d->ctrl[row] = controls._cached_control(row, 0);
    }
    std::cout << "CTRL: " << "\n" << controls._cached_control << std::endl;
    ctrl_buffer.emplace_back(controls._cached_control);
#endif
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::set_instance(MyController<T, state_size, ctrl_size> *myctrl)
{
    if constexpr(std::is_same<T, ILQR<state_size, ctrl_size>>::value)
    {
        my_ctrl_ilqr = myctrl;
    }
    else if constexpr(std::is_same<T, MPPI<state_size, ctrl_size>>::value)
    {
        my_ctrl_mppi = myctrl;
    }
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size,ctrl_size>::callback_wrapper(const mjModel *m, mjData *d)
{
    if constexpr(std::is_same<T, ILQR<state_size, ctrl_size>>::value)
    {
        my_ctrl_ilqr->controller();
    }
    else if constexpr(std::is_same<T, MPPI<state_size, ctrl_size>>::value)
    {
        my_ctrl_mppi->controller();
    }
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::dummy_controller(const mjModel *m, mjData *d)
{

}

template class MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>;
template class MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>;
