
#include <iostream>
#include <chrono>
#include <random>
#include "controller.h"
#include "../utilities/finite_diff.h"
#include "../utilities/internal_types.h"
#include "cost_function.h"
#include "simulation_params.h"
#include "../utilities/basic_math.h"

using namespace SimulationParameters;
static MyController<MPPI<n_jvel + n_jpos, n_ctrl>, n_jvel + n_jpos, n_ctrl> *my_ctrl_mppi;
static MyController<ILQR<n_jvel + n_jpos, n_ctrl>, n_jvel + n_jpos, n_ctrl> *my_ctrl_ilqr ;
static bool first = true;
static int _mark;

#define myFREESTACK d->pstack = _mark;

namespace
{
    //TODO: use jac_vel
    template<int state_size, int num_bodies>
    Eigen::Matrix<double, 3, num_bodies> compute_cartesian_ee_velocity(mjData *d, const mjModel* m)
    {
        double pos[3] = {d->xpos[0], d->xpos[1], d->xpos[2]};

        _mark = d->pstack;
        mjtNum * jac_pos = mj_stackAlloc(d, m->nv * 3);
        mjtNum * jac_vel = mj_stackAlloc(d, m->nv * 3);

        auto ee_num = m->nbody-1;
        mj_jac(m, d, jac_pos, nullptr, pos, ee_num);

        Eigen::Matrix<double, state_size/2, 1> joint_vel;
        Eigen::Matrix<double, 3, num_bodies> jacobian_pos;

        for(auto row = 0; row < m->nv; ++ row)
            joint_vel(row, 0) = d->qvel[row];

        for(auto row = 0; row < 3; ++row)
            for(auto body = 0; body < m->nbody; ++ body)
                jacobian_pos(row, body) = jac_pos[3*row+body];

        myFREESTACK;
        return jacobian_pos;
    }
}

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

    if (first) {
        first = not first;
    }
//    ctrl_buffer.emplace_back(controls._cached_control);
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


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::fill_control_buffer(const std::vector<Eigen::Matrix<double, ctrl_size, 1>> buffer)
{
    ctrl_buffer = buffer;
}


template class MyController<MPPI<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>;
template class MyController<ILQR<n_jpos + n_jvel, n_ctrl>, n_jpos + n_jvel, n_ctrl>;
