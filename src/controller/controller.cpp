
#include <iostream>
#include <chrono>
#include <random>
#include "controller.h"
#include "cost_function.h"
#include "ilqr.h"
#include "../third_party/FIC/fic.h"
#include "../parameters/simulation_params.h"
#include "../utilities/mujoco_utils.h"
#include "ilqr.h"
#include "mppi_ddp.h"

using namespace MujocoUtils;
using namespace SimulationParameters;
static MyController<ILQR, n_jvel + n_jpos, n_ctrl> *my_ctrl_ilqr;
static MyController<MPPIDDP, n_jvel + n_jpos, n_ctrl> *my_ctrl_mppi_ddp;
static MyController<uoe::FICController, state_size, n_ctrl> *my_ctrl_fic;
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
MyController<T, state_size, ctrl_size>::MyController(const mjModel *m, mjData *d, const T& controls, const bool comp_gravity) :
controls(controls), _m(m), _d(d), m_comp_gravity(comp_gravity), m_grav_comp(_d->qfrc_bias, _m->nu, 1)
{
    ctrl_buffer.assign(10, CtrlVector::Zero());
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::controller()
{
//    mju_copy(_d->qfrc_applied, _d->qfrc_bias, _m->nv);
    set_control_data(_d, controls.cached_control, _m);
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::set_instance(MyController<T, state_size, ctrl_size> *myctrl)
{
    if constexpr(std::is_same<T, ILQR>::value)
    {
        my_ctrl_ilqr = myctrl;
    }
    else if constexpr(std::is_same<T, MPPIDDP>::value)
    {
        my_ctrl_mppi_ddp = myctrl;
    }
    else if constexpr(std::is_same<T, uoe::FICController>::value)
    {
        my_ctrl_fic = myctrl;
    }

}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size,ctrl_size>::callback_wrapper(const mjModel *m, mjData *d)
{
    if constexpr(std::is_same<T, ILQR>::value)
    {
        my_ctrl_ilqr->controller();
    }
    else if constexpr(std::is_same<T, MPPIDDP>::value)
    {
        my_ctrl_mppi_ddp->controller();
    }
    else if constexpr(std::is_same<T, uoe::FICController>::value)
    {
        my_ctrl_fic->controller();
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


template class MyController<ILQR, state_size, n_ctrl>;
template class MyController<MPPIDDP, state_size, n_ctrl>;
template class MyController<uoe::FICController, state_size, n_ctrl>;