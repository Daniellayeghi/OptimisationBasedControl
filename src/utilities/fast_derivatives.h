#ifndef OPTCONTROL_MUJOCO_FAST_DERIVATIVES_H
#define OPTCONTROL_MUJOCO_FAST_DERIVATIVES_H

#include <vector>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include "../utilities/mujoco_utils.h"


using namespace SimulationParameters;


enum class WRT {POS = 0, VEl = 2, ACC = 3, CTRL = 4};

template<typename F, typename  T>
struct MJDerivativeParams{
    Eigen::Map<Eigen::Matrix<T, -1, 1>>& m_wrt;
    F& m_func;
    const T m_eps = 1e-6;
    WRT m_wrt_id = WRT::POS;
};


struct MJDataEig
{
    explicit MJDataEig(const mjModel* m) :
    m_m(m),
    m_d(mj_makeData(m)),
    m_ctrl(m_d->ctrl, m_m->nu),
    m_pos(m_d->qpos, m_m->nq),
    m_vel(m_d->qvel, m_m->nv),
    m_acc(m_d->qacc, m_m->nv),
    m_sens(m_d->sensordata, m_m->nsensordata)
    {}


    void set_state(const PosVector& pos, const VelVector& vel) {m_pos = pos; m_vel = vel;}
    void set_ctrl(const CtrlVector& ctrl){m_ctrl = ctrl;}

    const mjModel* m_m;
    mjData* m_d;
    Eigen::Map<Eigen::VectorXd> m_ctrl;
    Eigen::Map<Eigen::VectorXd> m_pos;
    Eigen::Map<Eigen::VectorXd> m_vel;
    Eigen::Map<Eigen::VectorXd> m_acc;
    Eigen::Map<Eigen::VectorXd> m_sens;
};


template <typename F, typename T>
class MJDynDerivative
{
public:
    explicit MJDynDerivative(const mjModel* m): m_m(m), m_ed(m), m_params{m_ed.m_ctrl, mj_step}

    {
        m_sens_res = Eigen::MatrixXd(m_m->nsensordata, m_params.m_wrt.size());
        m_dyn_res = Eigen::MatrixXd(m_m->nq +  m_m->nv * 2, m_params.m_wrt.size());
        m_pert = Eigen::VectorXd::Ones(m_params.m_wrt.size()) * m_params.m_eps;
    };


    const Eigen::MatrixXd& dyn_derivative(const MJDataEig& ed)
    {
        MujocoUtils::copy_data(m_m, ed.m_d, m_ed.m_d);
        for (int i = 0; i < m_params.m_wrt.size(); ++i)

        {
            perturb(i);
            m_params.m_func(m_m, m_ed.m_d);
            m_dyn_res.block(0, i, n_jpos, 1) =  (m_ed.m_pos - ed.m_pos) / m_params.m_eps;
            m_dyn_res.block(n_jpos, i, n_jvel, 1) =  (m_ed.m_vel - ed.m_vel) / m_params.m_eps;
            m_dyn_res.block(state_size, i, n_jvel, 1) =  (m_ed.m_acc - ed.m_acc) / m_params.m_eps;
            MujocoUtils::copy_data(m_m, ed.m_d, m_ed.m_d);
        }
        return m_dyn_res;
    };


    const Eigen::MatrixXd& sens_derivative(const MJDataEig& ed)
    {
        MujocoUtils::copy_data(m_m, ed.m_d, m_ed.m_d);
        for (int i = 0; i < m_params.m_wrt.size(); ++i)
        {
            perturb(i);
            m_params.m_func(m_m, m_ed.m_d);
            m_sens_res.col(i) = (m_ed.m_sens - ed.m_sens)/m_params.m_eps;
            MujocoUtils::copy_data(m_m, ed.m_d, m_ed.m_d);
        }
        return m_sens_res;
    };


private:

    // Deal with free and ball joints
    void perturb(const int idx)
    {
        if (m_params.m_wrt_id == WRT::POS)
        {
            // get quaternion address if applicable
            const auto jid = m_m->dof_jntid[idx];
            int quatadr = -1, dofpos = 0;
            if(m_m->jnt_type[jid] == mjJNT_BALL and m_params.m_wrt_id == WRT::POS)
            {
                quatadr = m_m->jnt_qposadr[jid];
                dofpos = idx - m_m->jnt_dofadr[jid];
            }
            else if(m_m->jnt_type[jid] == mjJNT_FREE && idx >= m_m->jnt_dofadr[jid]+3 and m_params.m_wrt_id == WRT::POS)
            {
                quatadr = m_m->jnt_qposadr[jid] + 3;
                dofpos = idx - m_m->jnt_dofadr[jid] - 3;
            }

            // apply quaternion or simple perturbation
            if(quatadr>=0 and m_params.m_wrt_id == WRT::POS)
            {
                mjtNum angvel[3] = {0,0,0};
                angvel[dofpos] = m_params.m_eps;
                mju_quatIntegrate(m_params.m_wrt.data()+quatadr, angvel, 1);
            }
            else
                m_params.m_wrt.data()[m_m->jnt_qposadr[jid] + idx - m_m->jnt_dofadr[jid]] += m_params.m_eps;
        }else{
            m_params.m_wrt(idx) = m_params.m_wrt(idx) + m_params.m_eps;
        }
    }


public:
    MJDerivativeParams<F, T> m_params;
    MJDataEig m_ed;

private:
    Eigen::VectorXd m_pert;
    Eigen::MatrixXd m_dyn_res;
    Eigen::MatrixXd m_sens_res;
    const mjModel* m_m;
};


#endif //OPTCONTROL_MUJOCO_FAST_DERIVATIVES_H
