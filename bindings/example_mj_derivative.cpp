#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <mujoco/mujoco.h>
#include <iostream>
#include <vector>
#include <mujoco/mjdata.h>
#include <Eigen/Core>

namespace
{
    void copy_data(const mjModel *model, const mjData *data_src, mjData *data_cp)
    {
        data_cp->time = data_src->time;
        mju_copy(data_cp->qpos, data_src->qpos, model->nq);
        mju_copy(data_cp->qvel, data_src->qvel, model->nv);
        mju_copy(data_cp->qacc, data_src->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data_src->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data_src->xfrc_applied, 6 * model->nbody);
        mju_copy(data_cp->ctrl, data_src->ctrl, model->nu);
    }

    enum class WRT {POS = 0, VEL = 2, ACC = 3, CTRL = 4};
}

struct MJDerivativeParams{
    using f_ptr = void(const mjModel*, mjData*);
    Eigen::Map<Eigen::Matrix<double, -1, 1>>& m_wrt;
    f_ptr& m_func;
    const double m_eps = 1e-6;
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

    void set_state(const Eigen::VectorXd& pos, const Eigen::VectorXd& vel) {m_pos = pos; m_vel = vel;}
    void set_ctrl(const Eigen::VectorXd& ctrl){m_ctrl = ctrl;}


    const mjModel* m_m;
    mjData* m_d;
    Eigen::Map<Eigen::VectorXd> m_ctrl;
    Eigen::Map<Eigen::VectorXd> m_pos;
    Eigen::Map<Eigen::VectorXd> m_vel;
    Eigen::Map<Eigen::VectorXd> m_acc;
    Eigen::Map<Eigen::VectorXd> m_sens;
};


class MJDerivative
{
public:
    explicit MJDerivative(const mjModel* m): m_m(m), m_ed(m), m_params{m_ed.m_ctrl, mj_step}

    {
        m_sens_res = Eigen::MatrixXd(m_m->nsensordata, m_params.m_wrt.size());
        m_dyn_res = Eigen::MatrixXd(m_m->nq +  m_m->nv * 2, m_params.m_wrt.size());
        m_pert = Eigen::VectorXd::Ones(m_params.m_wrt.size()) * m_params.m_eps;
    };


    const Eigen::MatrixXd& dyn_derivative(const MJDataEig& ed, const WRT wrt)
    {
        select_ptr(wrt);
        copy_data(m_m, ed.m_d, m_ed.m_d);
        for (int i = 0; i < m_params.m_wrt.size(); ++i)

        {
            perturb(i);
            m_params.m_func(m_m, m_ed.m_d);
            m_dyn_res.block(0, i, m_m->nq, 1) =  (m_ed.m_pos - ed.m_pos) / m_params.m_eps;
            m_dyn_res.block(m_m->nq, i, m_m->nv, 1) =  (m_ed.m_vel - ed.m_vel) / m_params.m_eps;
            m_dyn_res.block(m_m->nq + m_m->nv, i, m_m->nv, 1) =  (m_ed.m_acc - ed.m_acc) / m_params.m_eps;
            copy_data(m_m, ed.m_d, m_ed.m_d);
        }
        return m_dyn_res;
    };


    const Eigen::MatrixXd& sens_derivative(const MJDataEig& ed, const WRT wrt)
    {
        select_ptr(wrt);
        copy_data(m_m, ed.m_d, m_ed.m_d);
        for (int i = 0; i < m_params.m_wrt.size(); ++i)
        {
            perturb(i);
            m_params.m_func(m_m, m_ed.m_d);
            m_sens_res.col(i) = (m_ed.m_sens - ed.m_sens)/m_params.m_eps;
            copy_data(m_m, ed.m_d, m_ed.m_d);
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
            if(m_m->jnt_type[jid] == mjJNT_BALL)
            {
                quatadr = m_m->jnt_qposadr[jid];
                dofpos = idx - m_m->jnt_dofadr[jid];
            }
            else if(m_m->jnt_type[jid] == mjJNT_FREE && idx >= m_m->jnt_dofadr[jid]+3)
            {
                quatadr = m_m->jnt_qposadr[jid] + 3;
                dofpos = idx - m_m->jnt_dofadr[jid] - 3;
            }

            // apply quaternion or simple perturbation
            if(quatadr>=0)
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


    void select_ptr(const WRT wrt)
    {
        switch (wrt)
        {
            case WRT::CTRL: m_params.m_wrt = m_ed.m_ctrl; break;
            case WRT::ACC: m_params.m_wrt = m_ed.m_acc; break;
            case WRT::VEL: m_params.m_wrt = m_ed.m_vel; break;
            case WRT::POS: m_params.m_wrt = m_ed.m_pos; break;
            default: std::cerr << "Cannot compute derivatives with respect to argument";
        }
    }

public:
    const mjModel* m_m;
    MJDataEig m_ed;
    MJDerivativeParams m_params;

private:
    Eigen::VectorXd m_pert;
    Eigen::MatrixXd m_dyn_res;
    Eigen::MatrixXd m_sens_res;
};

namespace py = pybind11;

PYBIND11_MODULE(mj_derivative, mj_d) {
    py::class_<MJDerivative>(mj_d, "MJDerivative")
            .def(py::init<const mjModel*>())
            .def("dyn_derivative", &MJDerivative::dyn_derivative)
            .def("sen_derivative", &MJDerivative::sens_derivative);

    py::class_<MJDataEig>(mj_d, "MJDataEig")
            .def(py::init<const mjModel*>())
            .def("set_state", &MJDataEig::set_state)
            .def("set_ctrl", &MJDataEig::set_ctrl);

#ifdef VERSION_INFO
    mj_d.attr("__version__") = VERSION_INFO;
#else
    mj_d.attr("__version__") = "dev";
#endif
}