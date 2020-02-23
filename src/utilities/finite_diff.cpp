#include <iostream>
#include "finite_diff.h"

using Mat33 = Eigen::Matrix<mjtNum, 3, 3>;

FiniteDifference::FiniteDifference(const mjModel* m) : _m(m)
{
    _d_cp = mj_makeData(m);
    _f_du  = (mjtNum*) mju_malloc(6*sizeof(mjtNum)*_m->nv*_m->nv);
    _wrt[WithRespectTo::ACC] = _d_cp->qacc;  _skip[WithRespectTo::ACC] = mjtStage::mjSTAGE_VEL;
    _wrt[WithRespectTo::VEL] = _d_cp->qvel;  _skip[WithRespectTo::VEL] = mjtStage::mjSTAGE_POS;
    _wrt[WithRespectTo::POS] = _d_cp->qpos;  _skip[WithRespectTo::POS] = mjtStage::mjSTAGE_NONE;
    _wrt[WithRespectTo::CTRL] = _d_cp->ctrl; _skip[WithRespectTo::CTRL] = mjtStage::mjSTAGE_VEL;
}


FiniteDifference::~FiniteDifference()
{
    mju_free(_f_du);
    mj_deleteData(_d_cp);
}


void FiniteDifference::f_u(mjData *d, mjtNum *wrt)
{
    Mat33 result;
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(_d_cp, _m->nv);

    copy_state(d);
#ifdef NDEBUG
    std::cout << "FD addr " << _m << std::endl;
#endif
    mj_forward(_m, _d_cp);

    // extra solver iterations to improve warmstart (qacc) at center point
    for(int rep = 1; rep < 3; ++rep)
        mj_forwardSkip(_m, _d_cp, mjSTAGE_VEL, 1);

    mjtNum* output = _d_cp->qacc;

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, _m->nv);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = wrt;
    const mjtNum* original = d->ctrl;

    first_order_forward_diff(target, original, output, center,result);
    mjFREESTACK
}


void FiniteDifference::first_order_forward_diff(mjtNum *target, const mjtNum *original,
                                                const mjtNum* output, const mjtNum* center, Mat33& result)
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
        {
            // The output of the system is w.r.t the x_dd of the 3 DOF. target which indexes on the outer loop
            // is u w.r.t of the 3DOF... This loop computes columns of the Jacobian, outer loop fills rows.
            result(i, j) = (output[j] - center[j])/eps;
        }
    }
    std::cout << "Printing Jacobian Matrix" << "\n";
    std::cout << result << "\n";
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
