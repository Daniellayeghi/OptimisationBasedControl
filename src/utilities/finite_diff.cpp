#include <iostream>
#include "finite_diff.h"
#include "internal_types.h"
#include "../src/controller/simulation_params.h"

static int _mark = 0;
#define myFREESTACK   _d_cp->pstack = _mark;

namespace
{
    template<int state_size, int ctrl_size>
    mjtNum* select_original_ptr(typename FiniteDifference<state_size, ctrl_size>::WithRespectTo wrt, mjData* d)
    {
        switch (wrt)
        {
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::CTRL: return d->ctrl;
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::FRC:  return d->qfrc_applied;
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::ACC:  return d->qacc;
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::VEL:  return d->qvel;
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::POS:  return d->qpos;
        }
    }


    template<int state_size, int ctrl_size>
    inline mjtStage skip_stage(typename FiniteDifference<state_size, ctrl_size>::WithRespectTo wrt)
    {
        switch (wrt)
        {
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::CTRL:
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::FRC:
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::ACC:  return mjtStage::mjSTAGE_VEL;
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::VEL:  return mjtStage::mjSTAGE_POS;
            case FiniteDifference<state_size, ctrl_size>::WithRespectTo::POS:  return mjtStage::mjSTAGE_NONE;
        }
    }

    template<typename T>
    inline void copy_state(const mjModel * model, const mjData *d, T* _d_cp)
    {
        _d_cp->time = d->time;
        mju_copy(_d_cp->qpos, d->qpos, model->nq);
        mju_copy(_d_cp->qvel, d->qvel, model->nv);
        mju_copy(_d_cp->qacc, d->qacc, model->nv);
        mju_copy(_d_cp->qacc_warmstart, d->qacc_warmstart, model->nv);
        mju_copy(_d_cp->qfrc_applied, d->qfrc_applied, model->nv);
        mju_copy(_d_cp->xfrc_applied, d->xfrc_applied, 6*model->nbody);
        mju_copy(_d_cp->ctrl, d->ctrl, model->nu);
    }
}


template<int state_size, int ctrl_size>
FiniteDifference<state_size, ctrl_size>::FiniteDifference(const mjModel* m) : _m(m)
{
    _d_cp = mj_makeData(m);
    _wrt[WithRespectTo::ACC] = _d_cp->qacc;
    _wrt[WithRespectTo::VEL] = _d_cp->qvel;
    _wrt[WithRespectTo::POS] = _d_cp->qpos;
    _wrt[WithRespectTo::CTRL] = _d_cp->ctrl;
    _wrt[WithRespectTo::FRC]  = _d_cp->qfrc_applied;
}


template<int state_size, int ctrl_size>
FiniteDifference<state_size, ctrl_size>::~FiniteDifference()
{
    mj_deleteData(_d_cp);
}


template<int state_size, int ctrl_size>
Eigen::Block<typename FiniteDifference<state_size, ctrl_size>::complete_jacobian, state_size, ctrl_size>
FiniteDifference<state_size, ctrl_size>::f_u()
{
    return _full_jacobian.template block<state_size, ctrl_size>(0,state_size);
}


template<int state_size, int ctrl_size>
Eigen::Block<typename FiniteDifference<state_size, ctrl_size>::complete_jacobian, state_size, state_size>
FiniteDifference<state_size, ctrl_size>::f_x()
{
    return _full_jacobian.template block<state_size, state_size>(0, 0);
}


template<int state_size, int ctrl_size>
void FiniteDifference<state_size, ctrl_size>::f_x_f_u(mjData *d)
{
    ctrl_jacobian ctrl_deriv     = diff_wrt_ctrl(d, _wrt[WithRespectTo::CTRL], WithRespectTo::CTRL);
    partial_state_jacobian f_pos = diff_wrt_state(d, _wrt[WithRespectTo::POS], WithRespectTo::POS);
    partial_state_jacobian f_vel = diff_wrt_state(d, _wrt[WithRespectTo::VEL], WithRespectTo::VEL);
    _full_jacobian << f_pos, f_vel, ctrl_deriv;
}


template<int state_size, int ctrl_size>
typename FiniteDifference<state_size, ctrl_size>::ctrl_jacobian
FiniteDifference<state_size, ctrl_size>::diff_wrt_ctrl(mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy)
{

    if (do_copy)
        copy_state(_m, d, _d_cp);

#ifdef NDEBUG
    std::cout << "FD addr " << _m << std::endl;
#endif

    mj_step(_m, _d_cp);
    auto skip = skip_stage<state_size, ctrl_size>(id);

//     extra solver iterations to improve warmstart (qacc) at center point
    for(int rep = 1; rep < 3; ++rep)
        mj_forwardSkip(_m, _d_cp, skip, 1);

    const mjtNum* output_pos = _d_cp->qpos;
    const mjtNum* output_vel = _d_cp->qvel;

    _mark = _d_cp->pstack;
    mjtNum * centre_pos = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(centre_pos, output_pos, _m->nv);

    mjtNum * centre_vel = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(centre_vel, output_vel, _m->nv);

    mjtNum * warmstart = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(warmstart, _d_cp->qacc_warmstart, _m->nv);

    mju_copy(_d_cp->qacc_warmstart, warmstart, _m->nv);

    copy_state(_m, d, _d_cp);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = wrt;
    const mjtNum* original = select_original_ptr<state_size, ctrl_size>(id, d);
    return finite_diff_wrt_ctrl(target, original, centre_pos, centre_vel, d, id);
}


template<int state_size, int ctrl_size>
typename FiniteDifference<state_size, ctrl_size>::partial_state_jacobian
FiniteDifference<state_size, ctrl_size>::diff_wrt_state(mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy)
{
    mj_step(_m, _d_cp);
    auto skip = skip_stage<state_size, ctrl_size>(id);

    // extra solver iterations to improve warmstart (qacc) at center point
    for(int rep = 1; rep < 3; ++rep)
        mj_forwardSkip(_m, _d_cp, skip, 1);

    const mjtNum* output_pos = _d_cp->qpos;
    const mjtNum* output_vel = _d_cp->qvel;

    _mark = _d_cp->pstack;
    mjtNum * centre_pos = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(centre_pos, output_pos, _m->nv);

    mjtNum * centre_vel = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(centre_vel, output_vel, _m->nv);

    mjtNum * warmstart = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(warmstart, _d_cp->qacc_warmstart, _m->nv);

    mju_copy(_d_cp->qacc_warmstart, warmstart, _m->nv);
    copy_state(_m, d, _d_cp);
    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = wrt;
    const mjtNum* original = select_original_ptr<state_size, ctrl_size>(id, d);
    return finite_diff_wrt_state(target, original, centre_pos, centre_vel, d, id);
}


template<int state_size, int ctrl_size>
typename FiniteDifference<state_size, ctrl_size>::ctrl_jacobian
FiniteDifference<state_size, ctrl_size>::finite_diff_wrt_ctrl(mjtNum *target,
                                                              const mjtNum *original,
                                                              const mjtNum *centre_pos,
                                                              const mjtNum *centre_vel,
                                                              const mjData *d,
                                                              const WithRespectTo id)
{

    auto row = (id == WithRespectTo::CTRL or id == WithRespectTo::FRC) ? _m->nu : _m->nv;
    ctrl_jacobian result;

    for(int i = 0; i < row; ++i)
    {
        // perturb selected target
        target[i] += eps;

        // evaluate dynamics, with center warmstart
        mj_step(_m, _d_cp);

        // compute column i of derivative 2
        for(int j = 0; j < ctrl_size; ++j)
        {
            // The output of the system is w.r.t the x_dd of the 3 DOF. target which indexes on the outer loop
            // is u w.r.t of the 3DOF... This loop computes columns of the Jacobian, outer loop fills rows.
            result(j, i) = (_d_cp->qpos[j] - centre_pos[j])/eps;
            result(j+row, i) = (_d_cp->qvel[j] - centre_vel[j])/eps;
        }
        // undo perturbation
        copy_state(_m, d, _d_cp);
        target[i] = original[i];
    }
#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    myFREESTACK
    return result;
}


template<int state_size, int ctrl_size>
typename FiniteDifference<state_size, ctrl_size>::partial_state_jacobian
FiniteDifference<state_size, ctrl_size>::finite_diff_wrt_state(mjtNum *target,
                                                               const mjtNum *original,
                                                               const mjtNum *centre_pos,
                                                               const mjtNum* centre_vel,
                                                               const mjData *d,
                                                               const WithRespectTo id)
{
    partial_state_jacobian result;
    auto row = _m->nv;

    for(int i = 0; i < row; i++)
    {
        // get joint id for this dof
        int jid = _m->dof_jntid[i];

        // get quaternion address and dof position within quaternion (-1: not in quaternion)
        int quatadr = -1, dofpos = 0;
        if(_m->jnt_type[jid] == mjJNT_BALL and id == WithRespectTo::POS)
        {
            quatadr = _m->jnt_qposadr[jid];
            dofpos = i - _m->jnt_dofadr[jid];
        }
        else if(_m->jnt_type[jid] == mjJNT_FREE && i >= _m->jnt_dofadr[jid]+3 and id == WithRespectTo::POS)
        {
            quatadr = _m->jnt_qposadr[jid] + 3;
            dofpos = i - _m->jnt_dofadr[jid] - 3;
        }

        // apply quaternion or simple perturbation
        if(quatadr>=0 and id == WithRespectTo::POS)
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = eps;
            mju_quatIntegrate(target+quatadr, angvel, 1);
        }
        else
            target[_m->jnt_qposadr[jid] + i - _m->jnt_dofadr[jid]] += eps;

        // evaluate dynamics, with center warmstart
        mj_step(_m, _d_cp);
        // compute column i of derivative 0
        for(int j = 0; j < row; j++)
        {
            result(j, i) = (_d_cp->qpos[j] - centre_pos[j])/eps;
            result(j+row, i) = (_d_cp->qvel[j] - centre_vel[j])/eps;
        }
        // undo perturbation
        copy_state(_m, d, _d_cp);
        mju_copy(target, original, _m->nq);
    }
#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    myFREESTACK
    return result;

}


using namespace SimulationParameters;
template class FiniteDifference<n_jpos + n_jvel, n_ctrl>;
//template class FiniteDifference<4, 2>;
