#include <mujoco/mujoco.h>
#include <Eigen/Core>

// https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf


using EigenMatrixXMap = Eigen::Map<Eigen::Matrix<double, -1, 1>>;

void copy_data(const mjModel *model, const mjData *data_src, mjData *data_cp) {
    data_cp->time = data_src->time;
    mju_copy(data_cp->qpos, data_src->qpos, model->nq);
    mju_copy(data_cp->qvel, data_src->qvel, model->nv);
    mju_copy(data_cp->qacc, data_src->qacc, model->nv);
    mju_copy(data_cp->qfrc_applied, data_src->qfrc_applied, model->nv);
    mju_copy(data_cp->xfrc_applied, data_src->xfrc_applied, 6 * model->nbody);
    mju_copy(data_cp->ctrl, data_src->ctrl, model->nu);
    mj_forward(model, data_cp);
}


enum class Wrt : int {
    State = 0, Ctrl = 1
};

enum class Mode : int {
    Fwd = 0, Inv = 1
};


enum class Order: int{
    First = 0, Second = 1
};


class MjDerivativeParams{
public:
    MjDerivativeParams(double eps, const Wrt wrt, const Mode mode, const Order order):
    m_eps(eps), m_wrt_id(wrt), m_mode_id(mode), m_order_id(order){};
    double m_eps = 1e-6;
    const Wrt m_wrt_id = Wrt::Ctrl;
    const Mode m_mode_id = Mode::Fwd;
    const Order m_order_id = Order::First;
};


struct MjDataVecView {
    MjDataVecView(const mjModel *m, mjData *d) :
            m_m(m),
            m_d(d),
            m_ctrl(m_d->ctrl, m_m->nu),
            m_qfrc_inverse(m_d->qfrc_inverse, m_m->nv),
            m_pos(m_d->qpos, m_m->nq),
            m_vel(m_d->qvel, m_m->nv),
            m_acc(m_d->qacc, m_m->nv),
            m_sens(m_d->sensordata, m_m->nsensordata) {}


    const mjModel* m_m;
    mjData* m_d;
    Eigen::Map <Eigen::VectorXd> m_ctrl;
    Eigen::Map <Eigen::VectorXd> m_qfrc_inverse;
    Eigen::Map <Eigen::VectorXd> m_pos;
    Eigen::Map <Eigen::VectorXd> m_vel;
    Eigen::Map <Eigen::VectorXd> m_acc;
    Eigen::Map <Eigen::VectorXd> m_sens;
};


class MjDerivative {
public:
    explicit MjDerivative(const mjModel* m, mjData*d, const MjDerivativeParams& params) :
            m_d(mj_makeData(m)), m_ed_internal(m, m_d), m_ed_external(m, d), m_m(m), m_wrts({m_ed_internal.m_ctrl}), m_params(params), m_func(mj_step) {
        m_wrts = select_ptr(m_params.m_wrt_id);
        m_func = select_mode(m_params.m_mode_id);
        auto cols = 0;
        for (const EigenMatrixXMap &wrt: m_wrts) { cols += wrt.size(); }
        m_sens_res = Eigen::MatrixXd(m_m->nsensordata, cols);
        if (m_params.m_mode_id == Mode::Fwd)
            m_func_res = Eigen::MatrixXd(m_m->nq + m_m->nv, cols);
        else
            m_func_res = Eigen::MatrixXd(m_m->nv, cols);

        if (m_params.m_order_id == Order::Second and m_params.m_mode_id == Mode::Fwd){
            if (m_params.m_wrt_id == Wrt::State)
                m_func_2nd_res = Eigen::MatrixXd(cols * (m_m->nq + m_m->nv), cols);
            else if (m_params.m_wrt_id == Wrt::Ctrl)
                m_func_2nd_res = Eigen::MatrixXd(cols * (m_m->nq + m_m->nv), cols);
        }
    };


    ~MjDerivative(){
        mj_deleteData(m_d);
    };


    const Eigen::MatrixXd &inv(){
        mjcb_control = [](const mjModel* m, mjData* d){};
        long col = 0;
        for(EigenMatrixXMap& wrt: m_wrts)
        {
            copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            for (int i = 0; i < wrt.size(); ++i) {
                perturb(i, wrt);
                m_func(m_m, m_ed_internal.m_d);
                m_func_res.block(0, col + i, m_m->nv, 1) = m_ed_internal.m_qfrc_inverse;
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                m_func(m_m, m_ed_internal.m_d);
                m_func_res.block(0, col + i, m_m->nv, 1) -= m_ed_internal.m_qfrc_inverse;
                m_func_res.block(0, col + i, m_m->nv, 1) /= m_params.m_eps;
            }
            col += wrt.size();
        }
        return m_func_res;
    };


    const Eigen::MatrixXd &fwd(){
        mjcb_control = [](const mjModel* m, mjData* d){};
        long col = 0;
        for(EigenMatrixXMap& wrt: m_wrts)
        {
            copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            for (int i = 0; i < wrt.size(); ++i) {
                // f(u + e)
                perturb(i, wrt);
                m_func(m_m, m_ed_internal.m_d);
                m_func_res.block(0, col + i, m_m->nq, 1) = m_ed_internal.m_pos;
                m_func_res.block(m_m->nq, col + i, m_m->nv, 1) = m_ed_internal.m_vel;
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                // f(u + e) - f(u) / eps
                m_func(m_m, m_ed_internal.m_d);
                m_func_res.block(0, col + i, m_m->nq, 1) -= m_ed_internal.m_pos;
                m_func_res.block(m_m->nq, col + i, m_m->nv, 1) -= m_ed_internal.m_vel;
                m_func_res.block(0, col + i, m_m->nq, 1) /= m_params.m_eps;
                m_func_res.block(m_m->nq, col + i, m_m->nv, 1) /= m_params.m_eps;

            }
            col += wrt.size();
        }

        copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
        return m_func_res;
    };


    const Eigen::MatrixXd &fwd_2nd()
    {
        mjcb_control = [](const mjModel* m, mjData* d){};
        int deriv_group = 0;

        auto nq = m_m->nq, nv = m_m->nv;
        auto nx = nq+nv;

        const auto eps = pow(m_params.m_eps, 2);
        auto deriv_size = 0; for (const EigenMatrixXMap &wrt: m_wrts) {
            deriv_size += wrt.size();
        }
        const int hess_group = int(m_func_2nd_res.rows() / deriv_size);
        copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

        for(EigenMatrixXMap& wrt: m_wrts) {
            const int d_size = wrt.size();
            for (int d1 = 0; d1 < wrt.size(); ++d1) {
                for (int d2 = 0; d2 < wrt.size(); ++d2) {
                    const int deriv_i = d1 + deriv_group; const int deriv_j = d2;
                    // f(arg + hi*ei + hjej)
                    perturb(d1, wrt);
                    perturb(d2, wrt);

                    m_func(m_m, m_ed_internal.m_d);

                    for(int q = 0; q < nq; ++q)
                        m_func_2nd_res(deriv_i + (q * d_size), deriv_j) = m_ed_internal.m_pos[q];

                    for(int v = 0; v < nv; ++v)
                    {
                        auto h_i = v + nq;
                        m_func_2nd_res(deriv_i + (h_i * d_size), deriv_j) = m_ed_internal.m_vel[v];
                    }

                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                    // f(arg + hi*ei + hj * ej) - f(arg + hi*ei)
                    perturb(d1, wrt);

                    m_func(m_m, m_ed_internal.m_d);

                    for(int q = 0; q < m_m->nq; ++q)
                        m_func_2nd_res(deriv_i + (q * d_size), deriv_j) -= m_ed_internal.m_pos[q];

                    for(int v = 0; v < nv; ++v)
                    {
                        auto h_i = v + nq;
                        m_func_2nd_res(deriv_i + (h_i * d_size), deriv_j) -= m_ed_internal.m_vel[v];
                    }

                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                    // f(arg + hi*ei + hj * ej) - f(arg + hi*ei) - f(arg + hj*ej)
                    perturb(d2, wrt);

                    m_func(m_m, m_ed_internal.m_d);

                    for(int q = 0; q < m_m->nq; ++q)
                        m_func_2nd_res(deriv_i + (q * d_size), deriv_j) -= m_ed_internal.m_pos[q];

                    for(int v = 0; v < m_m->nv; ++v)
                    {
                        auto h_i = v + nq;
                        m_func_2nd_res(deriv_i + (h_i * d_size), deriv_j) -= m_ed_internal.m_vel[v];
                    }

                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                    // f(arg + hi*ei + hj * ej) - f(arg + hi*ei) - f(arg + hj*ej) + f(arg)
                    m_func(m_m, m_ed_internal.m_d);

                    for(int q = 0; q < m_m->nq; ++q)
                    {
                        m_func_2nd_res(deriv_i + (q * d_size), deriv_j) += m_ed_internal.m_pos[q];
                        m_func_2nd_res(deriv_i + (q * d_size), deriv_j) /= eps;
                    }

                    for(int v = 0; v < m_m->nv; ++v)
                    {
                        auto h_i = v + nq;
                        m_func_2nd_res(deriv_i + (h_i * d_size), deriv_j) += m_ed_internal.m_vel[v];
                        m_func_2nd_res(deriv_i + (h_i * d_size), deriv_j) /= eps;
                    }

                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
                }
            }
            deriv_group += wrt.size();
        }
        return m_func_2nd_res;
    }

    const Eigen::MatrixXd &output(){
        if(m_params.m_mode_id == Mode::Fwd)
            return m_params.m_order_id == Order::First ? fwd() : fwd_2nd();
        else
            return inv();
    };


    const Eigen::MatrixXd &sensor() {
        mjcb_control = [](const mjModel* m, mjData* d){};
        long col = 0;
        for(EigenMatrixXMap& wrt: m_wrts)
        {
            copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            for (int i = 0; i < wrt.size(); ++i) {
                // f(x + e)
                perturb(i, wrt);
                mj_forward(m_m, m_ed_internal.m_d);
                m_sens_res.col(i + col) = (m_ed_internal.m_sens);
                // f(x + e) - f(x) / e
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
                mj_forward(m_m , m_ed_internal.m_d);
                m_sens_res.col(i + col) -= (m_ed_internal.m_sens);
                m_sens_res.col(i + col) /= m_params.m_eps;
            }
            col += wrt.size();
        }
        return m_sens_res;
    };

private:
    // Deal with free and ball joints
    void perturb(const int idx, EigenMatrixXMap& wrt) {
        if (wrt.data() >= m_ed_internal.m_pos.data() and wrt.data() <= &m_ed_internal.m_pos.data()[m_m->nq]) {
            // get quaternion address if applicable`
            const auto jid = m_m->dof_jntid[idx];
            int quatadr = -1, dofpos = 0;
            if (m_m->jnt_type[jid] == mjJNT_BALL) {
                quatadr = m_m->jnt_qposadr[jid];
                dofpos = idx - m_m->jnt_dofadr[jid];
            } else if (m_m->jnt_type[jid] == mjJNT_FREE && idx >= m_m->jnt_dofadr[jid] + 3) {
                quatadr = m_m->jnt_qposadr[jid] + 3;
                dofpos = idx - m_m->jnt_dofadr[jid] - 3;
            }

            // apply quaternion or simple perturbation
            if (quatadr >= 0) {
                mjtNum angvel[3] = {0, 0, 0};
                angvel[dofpos] = m_params.m_eps;
                mju_quatIntegrate(wrt.data() + quatadr, angvel, 1);
            } else
                wrt.data()[m_m->jnt_qposadr[jid] + idx - m_m->jnt_dofadr[jid]] += m_params.m_eps;
        } else {
            wrt(idx) = wrt(idx) + m_params.m_eps;
        }
    }


    std::vector<std::reference_wrapper<EigenMatrixXMap>> select_ptr(const Wrt wrt) {
        if(m_params.m_mode_id == Mode::Fwd)
            switch (wrt) {
                case Wrt::Ctrl:
                    return {m_ed_internal.m_ctrl};
                    break;
                case Wrt::State:
                    return{m_ed_internal.m_pos, m_ed_internal.m_vel};
                default:
                    return {m_ed_internal.m_ctrl};
            }
        else
            return{m_ed_internal.m_pos, m_ed_internal.m_vel, m_ed_internal.m_acc};
    }


    mjfGeneric select_mode(const Mode mode){
        switch (mode) {
            case Mode::Fwd:
                return mj_step;
                break;
            case Mode::Inv:
                return mj_inverse;
                break;
            default:
                return mj_step;
        }
    }


private:
    mjData* m_d;
    MjDataVecView m_ed_internal;
    const MjDataVecView m_ed_external;
    const mjModel *m_m;
    Eigen::MatrixXd m_func_res;
    Eigen::MatrixXd m_func_2nd_res;
    Eigen::MatrixXd m_sens_res;
    std::vector<std::reference_wrapper<EigenMatrixXMap>> m_wrts;
    const MjDerivativeParams m_params;
    mjfGeneric m_func;
    Eigen::VectorXd m_perts;
};
