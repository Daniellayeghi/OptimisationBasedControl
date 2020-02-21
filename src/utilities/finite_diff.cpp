#include <iostream>
#include "finite_diff.h"

FiniteDifference::FiniteDifference(const mjModel* m) : _m(m)
{
    _d_cp = mj_makeData(m);
    _f_du  = (mjtNum*) mju_malloc(6*sizeof(mjtNum)*_m->nv*_m->nv);
}


FiniteDifference::~FiniteDifference()
{
    mju_free(_f_du);
    mj_deleteData(_d_cp);
}


void FiniteDifference::f_u(const mjData *d)
{
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(_d_cp, _m->nv);

    copy_state(d);
    std::cout << "FD addr " << _m << std::endl;
    mj_forward(_m, _d_cp);

    // extra solver iterations to improve warmstart (qacc) at center point
    for(int rep = 1; rep < 3; ++rep)
        mj_forwardSkip(_m, _d_cp, mjSTAGE_VEL, 1);

    mjtNum* output = _d_cp->qacc;

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, _m->nv);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = _d_cp->qfrc_applied;
    const mjtNum* original = d->qfrc_applied;

    first_order_forward_diff(target, original, output, center);
}


void FiniteDifference::first_order_forward_diff(mjtNum *target, const mjtNum *original,
                                                const mjtNum* output, const mjtNum* center)
{
    mjtNum* warmstart = mj_stackAlloc(_d_cp, _m->nv);
    mju_copy(warmstart, _d_cp->qacc_warmstart, _m->nv);
    for(int i = 0; i < _m->nv; ++i)
    {
        // perturb selected target
        target[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(_d_cp->qacc_warmstart, warmstart, _m->nv);
        mj_forwardSkip(_m, _d_cp, mjSTAGE_VEL, 1);

        // undo perturbation
        target[i] = original[i];

        // compute column i of derivative 2
        for(int j = 0; j < _m->nv; ++j)
            _f_du[(3+2)*_m->nv*_m->nv + i + j*_m->nv] = (output[j] - center[j])/eps;
    }
//    std::cout << "Printing Jacobian Matrix" << "\n";
//    mju_printMat(_f_du, _m->nv, _m->nv);
}


void FiniteDifference::copy_state(const mjData *d)
{
    _d_cp->time = d->time;
    mju_copy(d->qpos, d->qpos, _m->nq);
    mju_copy(d->qvel, d->qvel, _m->nv);
    mju_copy(d->qacc, d->qacc, _m->nv);
    mju_copy(d->qacc_warmstart, d->qacc_warmstart, _m->nv);
    mju_copy(d->qfrc_applied, d->qfrc_applied, _m->nv);
    mju_copy(d->xfrc_applied, d->xfrc_applied, 6*_m->nbody);
    mju_copy(d->ctrl, d->ctrl, _m->nu);
}