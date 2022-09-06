#include <mujoco/mujoco.h>
#include <Eigen/Core>


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
    State = 0, Ctrl = 1, StateCtrl = 2, FullState = 3
};

enum class Mode : int {
    Fwd = 0, Inv = 1
};


class MjDerivativeParams{
public:
    MjDerivativeParams(double eps, const Wrt wrt, const Mode mode): m_eps(eps), m_mode_id(mode){};
    double m_eps = 1e-6;
    const Mode m_mode_id = Mode::Fwd;
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


    const mjModel *m_m;
    mjData *m_d;
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
    m_d(mj_makeData(m)), m_ed_internal(m, m_d), m_ed_external(m, d),
    m_m(m), m_wrts({m_ed_internal.m_ctrl}),

    m_fwd_out_res(Eigen::MatrixXd(m->nq + m->nv, m->nq + m->nv + m->nu)),
    m_inv_out_res(Eigen::MatrixXd(m->nv, m->nq + m->nv + m->nv)),
    m_fwd_sens_res(Eigen::MatrixXd(m->nsensordata, m->nq + m->nv + m->nu)),
    m_inv_sens_res(Eigen::MatrixXd(m->nsensordata, m->nq + m->nv + m->nv)),

    m_wrts_map({
        {Wrt::State, {m_ed_internal.m_pos, m_ed_internal.m_vel}},{Wrt::Ctrl, {m_ed_internal.m_ctrl}},
        {Wrt::StateCtrl, {m_ed_internal.m_pos, m_ed_internal.m_vel, m_ed_internal.m_ctrl}},
        {Wrt::FullState, {m_ed_internal.m_pos, m_ed_internal.m_vel, m_ed_internal.m_acc}}
    }),
    m_params(params), m_func(mj_step)
    {
        m_func = select_mode(m_params.m_mode_id);
        auto cols = 0; for (const EigenMatrixXMap& wrt: m_wrts){cols += wrt.size();}
        m_sens_res = Eigen::MatrixXd(m_m->nsensordata, cols);
        if (m_params.m_mode_id == Mode::Fwd)
            m_func_res = Eigen::MatrixXd(m_m->nq + m_m->nv, cols);
        else
            m_func_res = Eigen::MatrixXd(m_m->nv, cols);

        m_fwd_out_res = Eigen::MatrixXd(m->nq + m->nv, m->nq + m->nv + m->nu);
        m_inv_out_res = Eigen::MatrixXd(m->nv, m->nq + m->nv + m->nv);
        m_fwd_sens_res = Eigen::MatrixXd(m->nsensordata, m->nq + m->nv + m->nu);
        m_inv_sens_res = Eigen::MatrixXd(m->nsensordata, m->nq + m->nv + m->nv);
    };


    ~MjDerivative(){
        mj_deleteData(m_d);
    };


    const Eigen::MatrixXd &inv(const Wrt wrt_id){
        mjcb_control = [](const mjModel* m, mjData* d){};
        long col = 0;
        for(EigenMatrixXMap& wrt: m_wrts)
        {
            copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            for (int i = 0; i < wrt.size(); ++i) {
                perturb(i, wrt);
                m_func(m_m, m_ed_internal.m_d);
                m_func_res.block(0, col + i, m_m->nv, 1) = (m_ed_internal.m_qfrc_inverse - m_ed_external.m_qfrc_inverse) / m_params.m_eps;
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            }
            col += wrt.size();
        }
        return m_func_res;
    };


    const Eigen::MatrixXd &fwd(const Wrt wrt){
        mjcb_control = [](const mjModel* m, mjData* d){};
        long col = 0;
        for(EigenMatrixXMap& wrt: m_wrts)
        {
            copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            for (int i = 0; i < wrt.size(); ++i) {
                perturb(i, wrt);
                m_func(m_m, m_ed_internal.m_d);
                m_func_res.block(0, col + i, m_m->nq, 1) = (m_ed_internal.m_pos - m_ed_external.m_pos) / m_params.m_eps;
                m_func_res.block(m_m->nq, col + i, m_m->nv, 1) = (m_ed_internal.m_vel - m_ed_external.m_vel) / m_params.m_eps;
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            }
            col += wrt.size();
        }
        return m_func_res;
    };


    const Eigen::MatrixXd &output(const Wrt wrt){
        if(m_params.m_mode_id == Mode::Fwd)
            return fwd(wrt);
        else
            return inv(wrt);
    };


    const Eigen::MatrixXd &sensor(const Wrt) {
        mjcb_control = [](const mjModel* m, mjData* d){};
        long col = 0;
        for(EigenMatrixXMap& wrt: m_wrts)
        {
            copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            for (int i = 0; i < wrt.size(); ++i) {
                perturb(i, wrt);
                m_func(m_m, m_ed_internal.m_d);
                m_sens_res.col(i + col) = (m_ed_internal.m_sens - m_ed_external.m_sens) / m_params.m_eps;
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
            }
            col += wrt.size();
        }
        return m_sens_res;
    };

private:
    // Deal with free and ball joints
    void perturb(const int idx, EigenMatrixXMap& wrt) {
        if (&wrt == &m_ed_internal.m_pos) {
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
//    Eigen::Block<Eigen::MatrixXd> m_dfdx;
//    Eigen::Block<Eigen::MatrixXd> m_dfdu;
//    Eigen::Block<Eigen::MatrixXd> m_dfsdu;
//    Eigen::Block<Eigen::MatrixXd> m_dfsdx;
//    Eigen::Block<Eigen::MatrixXd> m_dfinvdx_full;
//    Eigen::Block<Eigen::MatrixXd> m_dfinvsdx_full;
    Eigen::MatrixXd m_func_res = Eigen::MatrixXd(m_m->nq + m->nv, m->nq + m->nv + m->nu);
    Eigen::MatrixXd m_inv_out_res;
    Eigen::MatrixXd m_fwd_out_res;
    Eigen::MatrixXd m_fwd_sens_res;
    Eigen::MatrixXd m_inv_sens_res;
    Eigen::MatrixXd m_sens_res;
    std::vector<std::reference_wrapper<EigenMatrixXMap>> m_wrts;
    std::map<Wrt, std::vector<std::reference_wrapper<EigenMatrixXMap>>> m_wrts_map;
    const MjDerivativeParams m_params;
    mjfGeneric m_func;
};
