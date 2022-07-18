#ifndef OPTCONTROL_MUJOCO_FAST_DERIVATIVES_H
#define OPTCONTROL_MUJOCO_FAST_DERIVATIVES_H

#include <vector>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include "../utilities/mujoco_utils.h"


using namespace SimulationParameters;

template<typename F, typename  T>
struct MJDerivativeParams{
    Eigen::Map<Eigen::Matrix<T, -1, 1>>& m_wrt;
    const F& m_func;
    const T m_eps = 1e-6;
    const int m_nout = 0;
    const  int m_nin = 0;
};


struct MJDataEig
{
    explicit MJDataEig(const mjModel* m) :
    m_m(m),
    m_d(mj_makeData(m)),
    m_ctrl(m_d->ctrl, m_m->nu),
    m_pos(m_d->qpos, m_m->nq),
    m_vel(m_d->qvel, m_m->nv),
    m_acc(m_d->qacc, m_m->nv)
    {}


    void set_state(const PosVector& pos, const VelVector& vel)
    {
        m_pos = pos;
        m_vel = vel;
    }


    const mjModel* m_m;
    mjData* m_d;
    Eigen::Map<Eigen::VectorXd> m_ctrl;
    Eigen::Map<Eigen::VectorXd> m_pos;
    Eigen::Map<Eigen::VectorXd> m_vel;
    Eigen::Map<Eigen::VectorXd> m_acc;
};


template <typename F, typename T>
class MJDerivative
{
public:
    explicit MJDerivative(MJDerivativeParams<F, T>& params, const mjModel* m): m_params(params), m_m(m), m_ed(m)

    {
        m_res = Eigen::MatrixXd(m_params.m_nout, m_params.m_nin);
        m_pert = Eigen::VectorXd::Ones(m_params.m_nin) * m_params.m_eps;
    };


    const Eigen::MatrixXd& operator()(MJDataEig& ed)
    {
        MujocoUtils::copy_data(m_m, ed.m_d, m_ed.m_d);
        for (int i = 0; i < m_params.m_nin; ++i)
        {
            m_params.m_wrt(i) = m_params.m_wrt(i) + m_pert(i);
            m_params.m_func(m_m, ed.m_d);
            m_res.block(0, i, n_jpos, 1) =  (ed.m_pos - m_ed.m_pos) / m_params.m_eps;
            m_res.block(n_jpos, i, n_jvel, 1) =  (ed.m_vel - m_ed.m_vel) / m_params.m_eps;
            m_res.block(state_size, i, n_jvel, 1) =  (ed.m_acc - m_ed.m_acc) / m_params.m_eps;
            MujocoUtils::copy_data(m_m, m_ed.m_d, ed.m_d);
        }
        return m_res;
    };


    const MJDerivativeParams<F, T>& m_params;
    Eigen::VectorXd m_pert;
    Eigen::MatrixXd m_res;
    const mjModel* m_m;
    MJDataEig m_ed;
};

#endif //OPTCONTROL_MUJOCO_FAST_DERIVATIVES_H
