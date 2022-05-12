#include<iostream>
#include "mujoco.h"
#include "ilqr.h"
#include "../../src/utilities/mujoco_utils.h"

using namespace MujocoUtils;
using namespace SimulationParameters;

static void (*s_callback_ctrl)(const mjModel *, mjData *);
static auto step = [](const mjModel* m, mjData* d, mjfGeneric cbc){mjcb_control = cbc;  mj_step(m, d);};

ILQR::ILQR(FiniteDifference& fd,
           CostFunction& cf,
           ILQRParams& params,
           const mjModel * m,
           const mjData* d,
           const std::vector<CtrlVector>* init_u) :
        _fd(fd) , m_cf(cf), _m(m), m_params(params)
{
    _d_cp = mj_makeData(_m);
    m_regularizer.setIdentity();

    exp_cost_reduction.assign(m_params.simulation_time, 0.0);
    m_d_vector.assign(m_params.simulation_time + 1,{0, StateVector::Zero(), StateMatrix::Zero(),
                                                    CtrlVector::Zero(), CtrlMatrix::Zero(), CtrlStateMatrix::Zero(),
                                                    StateMatrix::Zero(), StateCtrlMatrix::Zero()});
    m_bp_vector.assign(m_params.simulation_time + 1, {CtrlVector::Zero(), CtrlStateMatrix::Zero()});
    m_x_traj_new.assign(m_params.simulation_time + 1, StateVector::Zero());
    m_x_traj.assign(m_params.simulation_time + 1, StateVector::Zero());
    m_u_traj_new.assign(m_params.simulation_time, CtrlVector::Zero());
    m_u_traj_cp.assign(m_params.simulation_time,CtrlVector::Zero());
    _covariance.assign(m_params.simulation_time, CtrlMatrix::Identity());
    _covariance_new.assign(m_params.simulation_time, CtrlMatrix::Identity());
    m_Qu_traj.assign(m_params.simulation_time, CtrlVector::Zero());
    m_Quu_traj.assign(m_params.simulation_time, CtrlMatrix::Zero());
    m_state_value.first.assign(m_params.simulation_time+1, StateVector::Zero());
    m_state_value.second.assign(m_params.simulation_time+1, 0);

    if (m_params.m_grav_comp)
        s_callback_ctrl = [](const mjModel* m, mjData *d){
            mju_copy(d->qfrc_applied, d->qfrc_bias, m->nu);
        };
    else
        s_callback_ctrl = [](const mjModel* m, mjData *d) {};


    copy_data(_m, d, _d_cp);
    fill_state_vector(_d_cp, m_x_traj.front(), _m);
    if(init_u == nullptr)
    {
        m_u_traj.assign(m_params.simulation_time, CtrlVector::Random() * 0);
    }else
    {
        m_u_traj = *init_u;
    }

    copy_data(m, d, _d_cp);
    for (int time = 0; time < m_params.simulation_time; ++time)
    {
        MujocoUtils::apply_ctrl_update_state(m_u_traj[time], m_x_traj[time+1], _d_cp, _m, s_callback_ctrl);
    }
    copy_data(m, d, _d_cp);


    m_backtrackers =  {{1.00000000e+00, 9.09090909e-01,
                        6.83013455e-01, 4.24097618e-01,
                        2.17629136e-01, 9.22959982e-02,
                        3.23491843e-02, 9.37040641e-03,
                        2.24320079e-03, 4.43805318e-04, 0.00000001}};
}


ILQR::~ILQR()
{
    mj_deleteData(_d_cp);
}


StateVector ILQR::Q_x(const int time, const StateVector& _v_x)
{
    return m_d_vector[time].lx + m_d_vector[time].fx.transpose() * _v_x ;
}


CtrlVector ILQR::Q_u(const int time,  const StateVector& _v_x)
{
    return m_d_vector[time].lu + m_d_vector[time].fu.transpose() * _v_x ;
}


StateMatrix ILQR::Q_xx(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].lxx + (m_d_vector[time].fx.transpose() * _v_xx * m_d_vector[time].fx);
}


CtrlStateMatrix ILQR::Q_ux(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].lux + (m_d_vector[time].fu.transpose() * (_v_xx) * m_d_vector[time].fx);
}


StateCtrlMatrix ILQR::Q_xu(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].fx.transpose() * (_v_xx) * m_d_vector[time].fu;
}


CtrlMatrix ILQR::Q_uu(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].luu + (m_d_vector[time].fu.transpose() * (_v_xx) * m_d_vector[time].fu);
}


StateMatrix ILQR::Q_xx_reg(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].lxx + (m_d_vector[time].fx.transpose() * (_v_xx + m_regularizer) * m_d_vector[time].fx);
}


CtrlStateMatrix ILQR::Q_ux_reg(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].lux + (m_d_vector[time].fu.transpose() * (_v_xx + m_regularizer) * m_d_vector[time].fx);
}


StateCtrlMatrix ILQR::Q_xu_reg(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].fx.transpose() * (_v_xx + m_regularizer) * m_d_vector[time].fu;
}


CtrlMatrix ILQR::Q_uu_reg(const int time, const StateMatrix& _v_xx)
{
    return m_d_vector[time].luu + (m_d_vector[time].fu.transpose() * (_v_xx + m_regularizer) * m_d_vector[time].fu);
}


void ILQR::forward_simulate(const mjData* d)
{
    _prev_total_cost = 0;
    copy_data(_m, d, _d_cp);
    for (auto time = 0; time < m_params.simulation_time; ++time)
    {
        set_control_data(_d_cp, m_u_traj[time], _m);
        _fd.f_x_f_u(_d_cp);
        m_d_vector[time].l = m_cf.running_cost(_d_cp);
        m_d_vector[time].lx = m_cf.L_x(_d_cp);
        m_d_vector[time].lxx = m_cf.L_xx(_d_cp);
        m_d_vector[time].lu = m_cf.L_u(_d_cp);
        m_d_vector[time].luu = m_cf.L_uu(_d_cp);
        m_d_vector[time].lux = m_cf.L_ux(_d_cp);
        m_d_vector[time].fx = _fd.f_x();
        m_d_vector[time].fu = _fd.f_u();
        _prev_total_cost += m_d_vector[time].l;
        step(_m, _d_cp, s_callback_ctrl);
    }
    _prev_total_cost += m_cf.terminal_cost(_d_cp);
    m_d_vector.back().l = m_cf.terminal_cost(_d_cp);
    m_d_vector.back().lx = m_cf.Lf_x(_d_cp);
    m_d_vector.back().lxx = m_cf.Lf_xx();
}


bool ILQR::minimal_grad()
{
    auto ctrl_k_ratio = 0.0;
    for(unsigned int iter = 0; iter < m_u_traj.size(); ++iter)
    {
        ctrl_k_ratio += (m_bp_vector[iter].ff_k.cwiseAbs().template cwiseProduct(
                        (m_u_traj[iter].cwiseAbs() + CtrlVector::Ones()).cwiseInverse()).maxCoeff());
    }
    if(ctrl_k_ratio/m_u_traj.size() < 1e-4 and m_regularizer(0, 0) < m_params.min_reg)
    {
        m_params.delta = std::min(m_params.delta_init, m_params.delta/m_params.delta_init);
        m_regularizer = m_regularizer * m_params.delta * (m_regularizer(0, 0) > m_params.min_reg);
        return true;
    }
    return false;
}


// TODO: make data const if you can
void ILQR::backward_pass()
{
    using hessian_matrix_t = Eigen::Matrix<double, state_size+n_ctrl, state_size+n_ctrl>;
    m_good_backpass = true;
    const auto max_iter = 1; auto iter = 0;
    auto non_pd_path = false;
    Eigen::Matrix<double, state_size, 1> V_x = m_d_vector.back().lx;
    Eigen::Matrix<double, state_size, state_size> V_xx = m_d_vector.back().lxx;
    hessian_matrix_t hessian;
//    printf("----------------------------------ILQR INFO-----------------------------------\n");
    do{
        for (auto time = m_params.simulation_time - 1; time >= 0; --time){
            ++iter;

            //General Approximations
            const StateVector Qx = Q_x(time, V_x); const CtrlVector Qu = Q_u(time, V_x); const StateCtrlMatrix Qxu = Q_xu(time, V_xx);
            const CtrlMatrix Quu = Q_uu(time, V_xx); const CtrlStateMatrix Qux = Q_ux(time , V_xx); const StateMatrix Qxx = Q_xx(time, V_xx);
            m_Quu_traj[time] = Quu; m_Qu_traj[time] = Qu;

            //Regularised Approximations
            const StateCtrlMatrix Qxu_reg = Q_xu_reg(time, V_xx); const CtrlMatrix Quu_reg = Q_uu_reg(time, V_xx);
            const CtrlStateMatrix Qux_reg = Q_ux_reg(time, V_xx); const StateMatrix Qxx_reg = Q_xx_reg(time, V_xx);

            const Eigen::LLT<CtrlMatrix> lltOfA(Quu_reg);
            const bool p = lltOfA.info() == Eigen::NumericalIssue;

            if (p) {
                non_pd_path = true; update_regularizer(true);
                m_good_backpass = (m_regularizer * m_params.delta)(0, 0) < 1e10;
                break;
            }


            //Compute the covariance from hessian
            hessian << Qxx_reg, Qxu_reg, Qux_reg, Quu_reg;
            const hessian_matrix_t hessian_inverse = hessian.llt().solve(
                    hessian_matrix_t::Identity()
            );

            const CtrlMatrix cov = hessian_inverse.block(
                    state_size, state_size, n_ctrl, n_ctrl
            );

            if (std::any_of(cov.data(), cov.data() + cov.size(), [](const double val) {return not std::isnan(val);}))
            {
                _covariance_new[time] = cov;
            } else {
                _covariance_new[time] = CtrlMatrix::Identity();
            }
//            printf("time %d Vx %f Vxx %f COV %f sum %f\n", time, V_x.sum(), V_xx.sum(), _covariance[time].trace(), m_u_traj[time].sum());

            // Compute the feedback and the feedforward gain
            m_bp_vector[time].fb_k = -1 * Quu_reg.colPivHouseholderQr().solve(Qux_reg);
            m_bp_vector[time].ff_k = -1 * Quu_reg.colPivHouseholderQr().solve(Qu);

            //Approximate value functions
            V_x = Qx + (m_bp_vector[time].fb_k.transpose() * Quu * (m_bp_vector[time].ff_k));
            V_x += m_bp_vector[time].fb_k.transpose() * Qu + Qux.transpose() * m_bp_vector[time].ff_k;
            V_xx = Qxx + m_bp_vector[time].fb_k.transpose() * Quu * m_bp_vector[time].fb_k;
            V_xx += m_bp_vector[time].fb_k.transpose() * Qux + Qux.transpose() * m_bp_vector[time].fb_k;
            V_xx = 0.5 * (V_xx + V_xx.transpose());
        }
    }while(non_pd_path and m_good_backpass and iter < max_iter);

}



void ILQR::temporal_average_covariance()
{
    auto iter = 0.0; auto weight_den = 0.0; CtrlMatrix weight_sum_num = CtrlMatrix::Zero();

    for(auto& cov :_covariance)
    {
        double weight = (m_params.simulation_time - iter)/m_params.simulation_time;
        weight_den += weight;
        weight_sum_num += (weight_sum_num + (weight * cov));
        cov = weight_sum_num/weight_den;
        ++iter;
    }
}


void ILQR::update_regularizer(const bool increase)
{

    if(increase)
    {
        m_params.delta = std::max(m_params.delta_init, m_params.delta * m_params.delta_init);
        auto reg_elem = std::max(m_regularizer(0, 0) * m_params.delta, m_params.min_reg);
        m_regularizer = StateMatrix::Identity() * reg_elem;
//        // All elements are equal hence the (0, 0) comparison
//        if ((m_regularizer * m_params.delta)(0, 0) > m_params.min_reg) {
//            m_regularizer = m_regularizer * m_params.delta;
//        } else {
//            m_regularizer = StateMatrix::Identity() * m_params.min_reg;
//        }
    }else
    {
        m_params.delta = std::min(1.0/m_params.delta_init, m_params.delta/m_params.delta_init);
        if ((m_regularizer * m_params.delta)(0, 0) > m_params.min_reg) {
            m_regularizer = m_regularizer * m_params.delta;
        } else {
            m_regularizer = StateMatrix::Zero();
        }
    }
}


double ILQR::compute_expected_cost(const double backtracker)
{
    double estimate_1st = 0 , estimate_2nd = 0;
    for (auto time = 0; time < m_params.simulation_time; ++time)
    {
        estimate_1st += (-backtracker*(m_bp_vector[time].ff_k.transpose() * m_Qu_traj[time])(0.0));
        estimate_2nd += (-backtracker*backtracker/2*(m_bp_vector[time].ff_k.transpose() * m_Quu_traj[time] * m_bp_vector[time].ff_k)(0.0));
    }
    return estimate_1st + estimate_2nd;
}


void ILQR::forward_pass(const mjData* d)
{
    static std::string status = "N";
    auto expected_cost_red = 0.0; auto new_total_cost = 0.0; auto cost_red_ratio = 0.0;

    //TODO Regularize the Quu inversion instead
    for (const auto &backtracker : m_backtrackers)
    {
        std::fill(m_u_traj_new.begin(), m_u_traj_new.end(), CtrlVector::Zero());
        copy_data(_m, d, _d_cp);
        m_x_traj_new.front() = m_x_traj.front();
        for (auto time = 0; time < m_params.simulation_time; ++time)
        {
            m_u_traj_new[time] =  m_u_traj[time] + (m_bp_vector[time].ff_k * backtracker) + m_bp_vector[time].fb_k * (m_x_traj_new[time] - m_x_traj[time]);
            clamp_control(m_u_traj_new[time], _m->actuator_ctrlrange);
            set_control_data(_d_cp, m_u_traj_new[time], _m);
            step(_m, _d_cp, s_callback_ctrl);
            fill_state_vector(_d_cp, m_x_traj_new[time + 1], _m);
        }

        // Check if backtracking needs to continue
        expected_cost_red = compute_expected_cost(backtracker);
        new_total_cost = m_cf.trajectory_running_cost(m_x_traj_new, m_u_traj_new);
        cost_red_ratio = (_prev_total_cost - new_total_cost)/expected_cost_red;

        // NOTE: Not doing this and updating regardless of the cost can lead to better performance!
        if(cost_red_ratio >= m_params.min_cost_red) {
            status = "Y";
//            update_regularizer(false);
            m_u_traj = m_u_traj_new;
            m_u_traj_cp = m_u_traj;
            m_x_traj = m_x_traj_new;
            _covariance = _covariance_new;
            printf("Cost = %f, Cost Diff = %f, Expected Diff = %f, Lambda = %f, Update = %s, last_position = %f\n",
                   _prev_total_cost, _prev_total_cost - new_total_cost, expected_cost_red, m_regularizer(0.0), status.c_str(), m_x_traj_new.front()(0, 0));
            break;
        }
        else if (cost_red_ratio < 0){
//            update_regularizer(true);
            status = "N";
        }
    }

    if (status == "N")
    {
        std::rotate(_covariance.begin(), _covariance.begin() + 1, _covariance.end());
        _covariance.back() = CtrlMatrix::Identity() * 0.15;
    }
}


void ILQR::control(const mjData* d, const bool skip)
{
    if (not skip) {
        m_params.delta = m_params.delta_init;
        m_regularizer.setIdentity();
        for (auto iteration = 0; iteration < m_params.iteration; ++iteration) {
            fill_state_vector(d, m_x_traj.front(), _m);
            forward_simulate(d);
            backward_pass();
            if (minimal_grad()) break;
            if (m_good_backpass) forward_pass(d);
            m_cf.m_u_prev = m_u_traj.front();
            cost.emplace_back(_prev_total_cost);
        }
    }
    else{
        std::rotate(_covariance.begin(), _covariance.begin() + 1, _covariance.end());
        _covariance.back() = CtrlMatrix::Identity() * 0.15;
    }

    cached_control = m_u_traj.front();
    m_cf.compute_value(m_u_traj, m_x_traj, m_state_value.second);
    m_state_value.first = m_x_traj;
    std::rotate(m_u_traj.begin(), m_u_traj.begin() + 1, m_u_traj.end());
    m_u_traj.back() = CtrlVector::Zero();
}
