#include "finite_diff.h"
#include "internal_types.h"

using namespace InternalTypes;

namespace
{
    mjtNum* select_original_ptr(const FiniteDifference::WithRespectTo wrt, mjData* d)
    {
        switch (wrt)
        {
            case FiniteDifference::WithRespectTo::CTRL: return d->ctrl;
            case FiniteDifference::WithRespectTo::FRC:  return d->qfrc_applied;
            case FiniteDifference::WithRespectTo::ACC:  return d->qacc;
            case FiniteDifference::WithRespectTo::VEL:  return d->qvel;
            case FiniteDifference::WithRespectTo::POS:  return d->qpos;
        }
    }


    mjtStage skip_stage(const FiniteDifference::WithRespectTo wrt)
    {
        switch (wrt)
        {
            case FiniteDifference::WithRespectTo::CTRL:
            case FiniteDifference::WithRespectTo::FRC:
            case FiniteDifference::WithRespectTo::ACC:  return mjtStage::mjSTAGE_VEL;
            case FiniteDifference::WithRespectTo::VEL:  return mjtStage::mjSTAGE_POS;
            case FiniteDifference::WithRespectTo::POS:  return mjtStage::mjSTAGE_NONE;
        }
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


Mat3x3 FiniteDifference::f_u(mjData *d)
{
    Mat3x3 result = differentiate(d, _wrt[WithRespectTo::CTRL], WithRespectTo::CTRL);
    return result;
}


Mat3x6 FiniteDifference::f_x(mjData *d)
{
    Mat3x6 result;
    Mat3x3 f_pos = differentiate(d, _wrt[WithRespectTo::POS], WithRespectTo::POS);
    Mat3x3 f_vel = differentiate(d, _wrt[WithRespectTo::VEL], WithRespectTo::VEL, false);

    result << f_pos, f_vel;
    return result;
}


void FiniteDifference::f_x_f_u(mjData *d)
{
    Mat3x3 ctrl_deriv  = differentiate(d, _wrt[WithRespectTo::CTRL], WithRespectTo::CTRL);
    Mat3x3 f_pos = differentiate(d, _wrt[WithRespectTo::POS], WithRespectTo::POS, false);
    Mat3x3 f_vel = differentiate(d, _wrt[WithRespectTo::VEL], WithRespectTo::VEL, false);
    _full_jacobian << f_pos, f_vel, ctrl_deriv;
}


Mat3x9& FiniteDifference::get_full_derivatives()
{
    return _full_jacobian;
}


Mat3x3 FiniteDifference::differentiate(mjData *d, mjtNum *wrt, const WithRespectTo id, bool do_copy)
{
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(_d_cp, _m->nv);

    if (do_copy)
        copy_state(d);

#ifdef NDEBUG
    std::cout << "FD addr " << _m << std::endl;
#endif
    mj_forward(_m, _d_cp);
    auto skip = skip_stage(id);

    // extra solver iterations to improve warmstart (qacc) at center point
    for(int rep = 1; rep < 3; ++rep)
        mj_forwardSkip(_m, _d_cp, skip, 1);

    mjtNum* output = _d_cp->qacc;

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, _m->nv);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = wrt;
    const mjtNum* original = select_original_ptr(id, d);

    if (id == WithRespectTo::POS)
        return first_order_forward_diff_positional(target, original, output, center, skip_stage(id));
    else
        return first_order_forward_diff_general(target, original, output, center, skip_stage(id));
}


Mat3x3 FiniteDifference::first_order_forward_diff_general(mjtNum *target, const mjtNum *original, const mjtNum* output,
                                                          const mjtNum* center, const mjtStage skip)
{
    Mat3x3 result;
    mjtNum* warmstart = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(warmstart, _d_cp->qacc_warmstart, _m->nv);

    for(int i = 0; i < _m->nv; ++i)
    {
        // perturb selected target
        target[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(_d_cp->qacc_warmstart, warmstart, _m->nv);
        mj_forwardSkip(_m, _d_cp, skip, 1);

        // undo perturbation
        target[i] = original[i];

        // compute column i of derivative 2
        for(int j = 0; j < _m->nv; ++j)
        {
            // The output of the system is w.r.t the x_dd of the 3 DOF. target which indexes on the outer loop
            // is u w.r.t of the 3DOF... This loop computes columns of the Jacobian, outer loop fills rows.
            result(i, j) = (output[j] - center[j])/eps;
        }
    }

#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    return result;
}


Mat3x3 FiniteDifference::first_order_forward_diff_positional(mjtNum *target, const mjtNum *original, const mjtNum* output,
                                                             const mjtNum* center, const mjtStage skip)
{
    Mat3x3 result;
    mjtNum* warmstart = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(warmstart, _d_cp->qacc_warmstart, _m->nv);

    for(int i = 0; i < _m->nv; i++)
    {
        // get joint id for this dof
        int jid = _m->dof_jntid[i];

        // get quaternion address and dof position within quaternion (-1: not in quaternion)
        int quatadr = -1, dofpos = 0;
        if(_m->jnt_type[jid] == mjJNT_BALL )
        {
            quatadr = _m->jnt_qposadr[jid];
            dofpos = i - _m->jnt_dofadr[jid];
        }
        else if(_m->jnt_type[jid] == mjJNT_FREE && i >= _m->jnt_dofadr[jid]+3)
        {
            quatadr = _m->jnt_qposadr[jid] + 3;
            dofpos = i - _m->jnt_dofadr[jid] - 3;
        }

        // apply quaternion or simple perturbation
        if( quatadr>=0 )
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = eps;
            mju_quatIntegrate(target+quatadr, angvel, 1);
        }
        else
            target[_m->jnt_qposadr[jid] + i - _m->jnt_dofadr[jid]] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(_d_cp->qacc_warmstart, warmstart, _m->nv);
        mj_forwardSkip(_m, _d_cp, skip, 1);


        // undo perturbation
        mju_copy(target, original, _m->nq);

        // compute column i of derivative 0
        for(int j = 0; j < _m->nv; j++) {
            result(i, j) = (output[j] - center[j]) / eps;
        }
    }

#if NDEBUG
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
#endif
    return result;
}


void FiniteDifference::copy_state(const mjData *d)
{
    _d_cp->time = d->time;
    mju_copy(_d_cp->qpos, d->qpos, _m->nq);
    mju_copy(_d_cp->qvel, d->qvel, _m->nv);
    mju_copy(_d_cp->qacc, d->qacc, _m->nv);
    mju_copy(_d_cp->qacc_warmstart, d->qacc_warmstart, _m->nv);
    mju_copy(_d_cp->qfrc_applied, d->qfrc_applied, _m->nv);
    mju_copy(_d_cp->xfrc_applied, d->xfrc_applied, 6*_m->nbody);
    mju_copy(_d_cp->ctrl, d->ctrl, _m->nu);
}


mjtNum * FiniteDifference::get_wrt(const WithRespectTo wrt)
{
    return _wrt[wrt];
}

