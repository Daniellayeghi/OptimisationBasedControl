
#include <chrono>
#include "controller.h"
#include "cost_function.h"
#include "ilqr.h"
#include "../third_party/FIC/fic.h"
#include "../utilities/mujoco_utils.h"
#include "mppi_ddp.h"
#include "par_mppi_ddp.h"

using namespace MujocoUtils;
using namespace SimulationParameters;
static MyController<ILQR, n_jvel + n_jpos, n_ctrl> *my_ctrl_ilqr;
static MyController<MPPIDDP, n_jvel + n_jpos, n_ctrl> *my_ctrl_mppi_ddp;
static MyController<MPPIDDPPar, n_jvel + n_jpos, n_ctrl> *my_ctrl_mppi_ddp_par;
static MyController<uoe::FICController, state_size, n_ctrl> *my_ctrl_fic;


// Gravity compensation assumes that all actuated bodies have to be compensated for and non actuated bodies
// are at the end of the model description
template<typename T, int state_size, int ctrl_size>
MyController<T, state_size, ctrl_size>::MyController(const mjModel *m, mjData *d, const T& controls, const bool comp_gravity) :
controls(controls), _m(m), _d(d), m_comp_gravity(comp_gravity), m_grav_force(_d->qfrc_bias, _m->nu, 1),
m_grav_comp(_d->qfrc_applied, _m->nu, 1)
{
    if(m_comp_gravity)
        m_gravity_setter = [&](){m_grav_comp = m_grav_force;};
    else
        m_gravity_setter = [&](){m_grav_comp = CtrlVector::Zero();};
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::controller()
{
    m_gravity_setter();
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
    else if constexpr(std::is_same<T, MPPIDDPPar>::value)
    {
        my_ctrl_mppi_ddp_par = myctrl;
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
    else if constexpr(std::is_same<T, MPPIDDPPar>::value)
    {
        my_ctrl_mppi_ddp_par->controller();
    }
    else if constexpr(std::is_same<T, uoe::FICController>::value)
    {
        my_ctrl_fic->controller();
    }
}


template<typename T, int state_size, int ctrl_size>
void MyController<T, state_size, ctrl_size>::dummy_controller(const mjModel *m, mjData *d) {}

template class MyController<ILQR, state_size, n_ctrl>;
template class MyController<MPPIDDP, state_size, n_ctrl>;
template class MyController<MPPIDDPPar, state_size, n_ctrl>;
template class MyController<uoe::FICController, state_size, n_ctrl>;