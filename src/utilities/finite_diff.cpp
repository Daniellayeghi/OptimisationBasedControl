#include <iostream>
#include "finite_diff.h"
#include "../utilities/mujoco_utils.h"
#include "../parameters/simulation_params.h"

static int _mark = 0;
#define myFREESTACK   _d_cp->pstack = _mark;

using namespace MujocoUtils;
namespace
{
    mjtNum* select_original_ptr(typename FiniteDifference::WithRespectTo wrt, const mjData* d)
    {
        switch (wrt)
        {
            case FiniteDifference::WithRespectTo::CTRL: return d->ctrl;
            case FiniteDifference::WithRespectTo::FRC:  return d->qfrc_applied;
            case FiniteDifference::WithRespectTo::ACC:  return d->qacc;
            case FiniteDifference::WithRespectTo::VEL:  return d->qvel;
            case FiniteDifference::WithRespectTo::POS:  return d->qpos;
        }
        return nullptr;
    }


    inline mjtStage skip_stage(typename FiniteDifference::WithRespectTo wrt)
    {
        switch (wrt)
        {
            case FiniteDifference::WithRespectTo::CTRL:
            case FiniteDifference::WithRespectTo::FRC:
            case FiniteDifference::WithRespectTo::ACC:  return mjtStage::mjSTAGE_VEL;
            case FiniteDifference::WithRespectTo::VEL:  return mjtStage::mjSTAGE_POS;
            case FiniteDifference::WithRespectTo::POS:  return mjtStage::mjSTAGE_NONE;
        }
        return mjtStage::mjSTAGE_NONE;
    }
}


FiniteDifference::FiniteDifference(const mjModel* m) : _m(m)
{
    _d_cp = mj_makeData(m);
    _wrt[WithRespectTo::ACC] = _d_cp->qacc;
    _wrt[WithRespectTo::VEL] = _d_cp->qvel;
    _wrt[WithRespectTo::POS] = _d_cp->qpos;
    _wrt[WithRespectTo::CTRL] = _d_cp->ctrl;
    _wrt[WithRespectTo::FRC]  = _d_cp->qfrc_applied;
}


FiniteDifference::~FiniteDifference()
{
    mj_deleteData(_d_cp);
}


Eigen::Block<typename FiniteDifference::complete_jacobian, state_size, n_ctrl>
FiniteDifference::f_u()
{
    return _full_jacobian.template block<state_size, n_ctrl>(0,state_size);
}


Eigen::Block<typename FiniteDifference::complete_jacobian, state_size, state_size>
FiniteDifference::f_x()
{
    return _full_jacobian.template block<state_size, state_size>(0, 0);
}


void FiniteDifference::f_x_f_u(mjData *d)
{
   diff_wrt(d, _wrt[WithRespectTo::CTRL], WithRespectTo::CTRL);
   diff_wrt(d, _wrt[WithRespectTo::POS], WithRespectTo::POS);
   diff_wrt(d, _wrt[WithRespectTo::VEL], WithRespectTo::VEL);
//   std::cout << ctrl_jac << std::endl;
   _full_jacobian << sp_jac, sv_jac, ctrl_jac;
}


GenericUtils::FastPair<mjtNum*, mjtNum *>
FiniteDifference::set_finite_diff_arguments(const mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy)
{

    if (do_copy)
        MujocoUtils::copy_data(_m, d, _d_cp);

    mj_step(_m, _d_cp);
    auto skip = skip_stage(id);

//     extra solver iterations to improve warmstart (qacc) at center point
    for(int rep = 1; rep < 3; ++rep)
        mj_forwardSkip(_m, _d_cp, skip, 1);

    const mjtNum* output_pos = _d_cp->qpos;
    const mjtNum* output_vel = _d_cp->qvel;

    _mark = _d_cp->pstack;
    mjtNum * centre_pos = mj_stackAlloc(_d_cp, _m->nq);
    mju_copy(centre_pos, output_pos, _m->nq);

    mjtNum * centre_vel = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(centre_vel, output_vel, _m->nv);

    mjtNum * warmstart = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(warmstart, _d_cp->qacc_warmstart, _m->nv);

    mju_copy(_d_cp->qacc_warmstart, warmstart, _m->nv);

    copy_data(_m, d, _d_cp);
    return {centre_pos, centre_vel};
}



void FiniteDifference::diff_wrt(const mjData *d, mjtNum *wrt, WithRespectTo id, bool do_copy)
{
    const auto [centre_pos, centre_vel] = set_finite_diff_arguments(d, wrt, id, do_copy);
    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = wrt;
    const mjtNum* original = select_original_ptr(id, d);
    const FDFuncArgs fd_args {target, original, centre_pos, centre_vel};

    switch(id) {
        case WithRespectTo::CTRL: ctrl_jac = finite_diff_wrt_ctrl(fd_args, d, id); break;
        case WithRespectTo::VEL: sv_jac = finite_diff_wrt_state_vel(fd_args, d, id); break;
        case WithRespectTo::POS: sp_jac = finite_diff_wrt_state_pos(fd_args, d, id); break;
    }
}


void FiniteDifference::perturb_target(mjtNum *target, const WithRespectTo id, const int state_iter)
{
    int jid = 0;
    if(id == WithRespectTo::POS)
    {
        jid = _m->dof_jntid[state_iter];
        // get joint id for this dof
        // get quaternion address and dof position within quaternion (-1: not in quaternion)
        int quatadr = -1, dofpos = 0;
        if(_m->jnt_type[jid] == mjJNT_BALL and id == WithRespectTo::POS)
        {
            quatadr = _m->jnt_qposadr[jid];
            dofpos = state_iter - _m->jnt_dofadr[jid];
        }
        else if(_m->jnt_type[jid] == mjJNT_FREE && state_iter >= _m->jnt_dofadr[jid]+3 and id == WithRespectTo::POS)
        {
            quatadr = _m->jnt_qposadr[jid] + 3;
            dofpos = state_iter - _m->jnt_dofadr[jid] - 3;
        }

        // apply quaternion or simple perturbation
        if(quatadr>=0 and id == WithRespectTo::POS)
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = eps;
            mju_quatIntegrate(target+quatadr, angvel, 1);
        }
        else
            target[_m->jnt_qposadr[jid] + state_iter - _m->jnt_dofadr[jid]] += eps;

    }else{
        target[state_iter] += eps;
    }

}


typename FiniteDifference::ctrl_jacobian
FiniteDifference::finite_diff_wrt_ctrl(const FDFuncArgs& fd_args, const mjData *d, const WithRespectTo id)
{
    static const auto row_ctrl = _m->nu;
    ctrl_jacobian result;
    double pos_diff, vel_diff = 0.0;

    for(int i = 0; i < row_ctrl; ++i)
    {
        perturb_target(fd_args.target, id, i);
        // evaluate dynamics, with center warmstart
        mj_step(_m, _d_cp);
        // compute column i of derivative 0
        for(int j = 0; j < _m->nq; ++j)
        {
            // The output of the system is w.r.t the x_dd of the 3 DOF. target which indexes on the outer loop
            // is u w.r.t of the 3DOF... This loop computes columns of the Jacobian, outer loop fills rows.
            pos_diff = _d_cp->qpos[j] - fd_args.centre_pos[j];
            result(j, i) = (pos_diff)/eps;
        }

        for(int j = _m->nq; j < _m->nv + _m->nq; ++j)
        {
            vel_diff = _d_cp->qvel[j - _m->nq] - fd_args.centre_vel[j - _m->nq];
            result(j, i) = vel_diff / eps;
        }
        // undo perturbation
        copy_data(_m, d, _d_cp);
        mju_copy(fd_args.target, fd_args.original, _m->nq);
    }
#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    myFREESTACK
    return result;
}

typename FiniteDifference::state_vel_jacobian
FiniteDifference::finite_diff_wrt_state_vel(const FDFuncArgs& fd_args, const mjData *d, const WithRespectTo id)
{
    state_vel_jacobian result;
    auto row = _m->nv;
    double pos_diff, vel_diff = 0.0;
    for(int i = 0; i < row; i++)
    {
        perturb_target(fd_args.target, id, i);
        // evaluate dynamics, with center warmstart
        mj_step(_m, _d_cp);
        // compute column i of derivative 0
        for(int j = 0; j < _m->nq; ++j)
        {
            // The output of the system is w.r.t the x_dd of the 3 DOF. target which indexes on the outer loop
            // is u w.r.t of the 3DOF... This loop computes columns of the Jacobian, outer loop fills rows.
            pos_diff = _d_cp->qpos[j] - fd_args.centre_pos[j];
            result(j, i) = (pos_diff)/eps;
        }

        for(int j = _m->nq; j < _m->nv + _m->nq; ++j)
        {
            vel_diff =  _d_cp->qvel[j - _m->nq] - fd_args.centre_vel[j - _m->nq];
            result(j, i) = vel_diff / eps;
        }
        // undo perturbation
        copy_data(_m, d, _d_cp);
        mju_copy(fd_args.target, fd_args.original, _m->nq);
    }
#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    myFREESTACK
    return result;

}


typename FiniteDifference::state_pos_jacobian
FiniteDifference::finite_diff_wrt_state_pos(const FDFuncArgs& fd_args, const mjData *d, const WithRespectTo id)
{
    state_pos_jacobian result;
    auto row = _m->nq;
    double pos_diff, vel_diff = 0.0;
    for(int i = 0; i < row; i++)
    {
        perturb_target(fd_args.target, id, i);
        // evaluate dynamics, with center warmstart
        mj_step(_m, _d_cp);
        // compute column i of derivative 0
        for(int j = 0; j < _m->nq; ++j)
        {
            // The output of the system is w.r.t the x_dd of the 3 DOF. target which indexes on the outer loop
            // is u w.r.t of the 3DOF... This loop computes columns of the Jacobian, outer loop fills rows.
            pos_diff = _d_cp->qpos[j] - fd_args.centre_pos[j];
            result(j, i) = (pos_diff)/eps;
        }

        for(int j = _m->nq; j < _m->nv + _m->nq; ++j)
        {
            vel_diff = _d_cp->qvel[j - _m->nq] - fd_args.centre_vel[j - _m->nq];
            result(j, i) = vel_diff / eps;
        }
        // undo perturbation
        copy_data(_m, d, _d_cp);
        mju_copy(fd_args.target, fd_args.original, _m->nq);
    }
#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    myFREESTACK
    return result;

}



