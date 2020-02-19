
#include <iostream>
#include <chrono>
#include "controller.h"

static MyController *my_ctrl;

using namespace std;

static int isforward = 1;
static float eps = 0.000001;
static int nwarmup = 3;


static void f_u(const mjModel* m, const mjData* dmain, mjData* d, mjtNum* f_du,  int id = 0)
{
    int nv = m->nv;
    // allocate stack space for result at center
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(d, nv);
    mjtNum* warmstart = mj_stackAlloc(d, nv);

    // prepare static schedule: range of derivative columns to be computed by this thread
    int chunk = (m->nv);
    int istart = id * chunk;
    int iend = mjMIN(istart + chunk, m->nv);

    // copy state and control from dmain to thread-specific d
    d->time = dmain->time;
    mju_copy(d->qpos, dmain->qpos, m->nq);
    mju_copy(d->qvel, dmain->qvel, m->nv);
    mju_copy(d->qacc, dmain->qacc, m->nv);
    mju_copy(d->qacc_warmstart, dmain->qacc_warmstart, m->nv);
    mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
    mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
    mju_copy(d->ctrl, dmain->ctrl, m->nu);

    std::cout << d << "\n";
    std::cout << dmain << "\n";

    // run full computation at center point (usually faster than copying dmain){
//    mj_forward(m, d);
//    std::cout << "Post FW" << "\n";

//     extra solver iterations to improve warmstart (qacc) at center point
    for( int rep=1; rep<nwarmup; rep++ )
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

    // select output from forward or inverse dynamics
    mjtNum* output = (isforward ? d->qacc : d->qfrc_inverse);

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, nv);
    mju_copy(warmstart, d->qacc_warmstart, nv);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = (isforward ? d->qfrc_applied : d->qacc);
    const mjtNum* original = (isforward ? dmain->qfrc_applied : dmain->qacc);

    // finite-difference over force or acceleration: skip = mjSTAGE_VEL
    for( int i=istart; i<iend; i++ )
    {
        // perturb selected target
        target[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

        // undo perturbation
        target[i] = original[i];

        // compute column i of derivative 2
        for( int j=0; j<nv; j++ )
            f_du[(3*isforward+2)*nv*nv + i + j*nv] = (output[j] - center[j])/eps;
    }

    std::cout << "Printing Jacobian Matrix" << "\n";
    mju_printMat(f_du, nv, nv);
}



static void f_uu(const mjModel* m, const mjData* dmain, mjData* d, mjtNum* f_duu, int id = 0)
{
    int nv = m->nv;

    // allocate stack space for result at center
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(d, nv);
    mjtNum* perturb_1 = mj_stackAlloc(d, nv);
    mjtNum* warmstart = mj_stackAlloc(d, nv);

    // prepare static schedule: range of derivative columns to be computed by this thread
    int chunk = (m->nv);
    int istart = id * chunk;
    int iend = mjMIN(istart + chunk, m->nv);

    // copy state and control from dmain to thread-specific d
    d->time = dmain->time;
    mju_copy(d->qpos, dmain->qpos, m->nq);
    mju_copy(d->qvel, dmain->qvel, m->nv);
    mju_copy(d->qacc, dmain->qacc, m->nv);
    mju_copy(d->qacc_warmstart, dmain->qacc_warmstart, m->nv);
    mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
    mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
    mju_copy(d->ctrl, dmain->ctrl, m->nu);

    // run full computation at center point (usually faster than copying dmain)
    mj_forward(m, d);

    // extra solver iterations to improve warmstart (qacc) at center point
    for( int rep=1; rep<nwarmup; rep++ )
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);


    // select output from forward or inverse dynamics
    mjtNum* output = d->qacc;

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, nv);
    mju_copy(warmstart, d->qacc_warmstart, nv);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target_1 = d->qfrc_applied;
    const mjtNum* original = dmain->qfrc_applied;

    // finite-difference over force or acceleration: skip = mjSTAGE_VEL
    for( int i=istart; i<iend; i++ )
    {
        // perturb selected target
        target_1[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);


        mju_copy(perturb_1, output, nv);
        target_1[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

        // undo perturbation
        target_1[i] = original[i];

        // compute column i of derivative 2
        for( int j=0; j<nv; j++ )
            f_duu[(3*isforward+2)*nv*nv + i + j*nv] = (output[j] - 2*perturb_1[j] + center[j])/(eps*eps);
    }
    std::cout << "Printing Hessian Matrix" << "\n";
    mju_printMat(f_duu, nv, nv);
}


MyController::MyController(const mjModel *m, const mjModel *m_cp, mjData *d, _mjData *d_cp) : _m(m), _m_cp(m_cp), _d(d), _d_cp(d_cp)
{
    _inertial_torque = mj_stackAlloc(_d, _m->nv);
    _constant_acc = mj_stackAlloc(d, m->nv);
    f_duu = (mjtNum*) mju_malloc(6*sizeof(mjtNum)*m->nv*m->nv);
    f_du  = (mjtNum*) mju_malloc(6*sizeof(mjtNum)*m->nv*m->nv);
    for (std::size_t row = 0; row < 3; ++row)
    {
        _constant_acc[row] = 0.4;
    }
}

MyController::~MyController()
{
    mju_free(f_duu);
    mju_free(f_du);
}


void MyController::controller()
{
    std::cout << _m << "\n";
    mj_mulM(_m, _d, _inertial_torque, _constant_acc);
//    f_u(_m_cp, _d, _d_cp, f_du);
//    _d->qfrc_applied[0] = _d->qfrc_bias[0] + _inertial_torque[0];
//    _d->qfrc_applied[1] = _d->qfrc_bias[1] + _inertial_torque[0];
//    _d->qfrc_applied[2] = _d->qfrc_bias[2] + _inertial_torque[0];
}


void MyController::set_instance(MyController *myctrl)
{
    my_ctrl = myctrl;
}


void MyController::callback_wrapper(const mjModel *m, mjData *d)
{
    my_ctrl->controller();
}