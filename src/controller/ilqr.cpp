#include<iostream>
#include "mujoco.h"
#include "ilqr.h"
#include "../../src/utilities/mujoco_utils.h"

using namespace MujocoUtils;
using namespace SimulationParameters;


ILQR::ILQR(FiniteDifference& fd,
                                  CostFunction& cf,
                                  ILQRParams& params,
                                  const mjModel * m,
                                  const mjData* d,
                                  const std::vector<CtrlVector>* init_u) :
_fd(fd) ,_cf(cf), _m(m), m_params(params)
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
        MujocoUtils::apply_ctrl_update_state(m_u_traj[time], m_x_traj[time+1], _d_cp, _m);
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


StateVector ILQR::Q_x(int time, StateVector& _v_x)
{
    return m_d_vector[time].lx + m_d_vector[time].fx.transpose() * _v_x ;
}


CtrlVector ILQR::Q_u(int time,  StateVector& _v_x)
{
    return m_d_vector[time].lu + m_d_vector[time].fu.transpose() * _v_x ;
}


StateMatrix ILQR::Q_xx(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lxx + (m_d_vector[time].fx.transpose() * _v_xx) * m_d_vector[time].fx;
}


CtrlStateMatrix ILQR::Q_ux(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lux + m_d_vector[time].fu.transpose() * (_v_xx) * m_d_vector[time].fx;
}


StateCtrlMatrix ILQR::Q_xu(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].fx.transpose() * (_v_xx) * m_d_vector[time].fu;
}


CtrlMatrix ILQR::Q_uu(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].luu + (m_d_vector[time].fu.transpose() * (_v_xx)) * (m_d_vector[time].fu);
}


StateMatrix ILQR::Q_xx_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lxx + (m_d_vector[time].fx.transpose() * _v_xx + m_regularizer) * m_d_vector[time].fx;
}


CtrlStateMatrix ILQR::Q_ux_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lux + (m_d_vector[time].fu.transpose() * (_v_xx + m_regularizer)) * m_d_vector[time].fx;
}


StateCtrlMatrix ILQR::Q_xu_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].fx.transpose() * (_v_xx + m_regularizer) * m_d_vector[time].fu;
}


CtrlMatrix ILQR::Q_uu_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].luu + (m_d_vector[time].fu.transpose() * (_v_xx + m_regularizer)) * (m_d_vector[time].fu);
}


void ILQR::forward_simulate(const mjData* d)
{
    _prev_total_cost = 0;
    copy_data(_m, d, _d_cp);
    for (auto time = 0; time < m_params.simulation_time; ++time)
    {
        set_control_data(_d_cp, m_u_traj[time], _m);
        _fd.f_x_f_u(_d_cp);
        m_d_vector[time].l = _cf.running_cost(_d_cp);
        m_d_vector[time].lx = _cf.L_x(_d_cp);
        m_d_vector[time].lxx = _cf.L_xx(_d_cp);
        m_d_vector[time].lu = _cf.L_u(_d_cp);
        m_d_vector[time].luu = _cf.L_uu(_d_cp);
        m_d_vector[time].lux = _cf.L_ux(_d_cp);
        m_d_vector[time].fx = _fd.f_x();
        m_d_vector[time].fu = _fd.f_u();
        _prev_total_cost += m_d_vector[time].l;
        mj_step(_m, _d_cp);
    }
    _prev_total_cost += _cf.terminal_cost(_d_cp);
    m_d_vector.back().l = _cf.terminal_cost(_d_cp);
    m_d_vector.back().lx = _cf.Lf_x(_d_cp);
    m_d_vector.back().lxx = _cf.Lf_xx();
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
    m_good_backpass = true;
    const auto max_iter = 20; auto iter = 0;
    auto non_pd_path = false;
    Eigen::Matrix<double, state_size, 1> V_x = m_d_vector.back().lx;
    Eigen::Matrix<double, state_size, state_size> V_xx = m_d_vector.back().lxx;
    Eigen::Matrix<double, state_size+n_ctrl, state_size+n_ctrl> hessian;

    do{
        for (auto time = m_params.simulation_time - 1; time >= 0; --time){
            //General Approximations
            const auto Qx = Q_x(time, V_x); const auto Qu = Q_u(time, V_x); const auto Qxu = Q_xu(time, V_xx);
            const auto Quu = Q_uu(time, V_xx); const auto Qux = Q_ux(time , V_xx); const auto Qxx = Q_xx(time, V_xx);
            hessian << Qxx, Qxu, Qux, Quu;
            m_Quu_traj[time] = Quu; m_Qu_traj[time] = Qu;

            //Regularised Approximations
            const auto Qxu_reg = Q_xu_reg(time, V_xx); const auto Quu_reg = Q_uu_reg(time, V_xx);
            const auto Qux_reg = Q_ux_reg(time, V_xx); const auto Qxx_reg = Q_xx_reg(time, V_xx);

            //Compute the covariance from hessian
            hessian << Qxx_reg, Qxu_reg, Qux_reg, Quu_reg;
            const auto hessian_inverse = hessian.llt().solve(
                    Eigen::Matrix<double, state_size+n_ctrl, state_size+n_ctrl>::Identity()
                    );

            const CtrlMatrix cov = hessian_inverse.block(
                    state_size, state_size, n_ctrl, n_ctrl
                    );

            Eigen::LLT<Eigen::MatrixXd> lltOfA(Quu_reg);
            auto p = lltOfA.info() == Eigen::NumericalIssue;
            if (p) {
                non_pd_path = true; update_regularizer(true);
                m_good_backpass = (m_regularizer * m_params.delta)(0, 0) < 1e10;
                break;
            }

            if (std::any_of(cov.data(), cov.data() + cov.size(), [](double val) {return not std::isnan(val);}))
            {
                _covariance_new[time] = cov;
            } else {
                _covariance_new[time] = CtrlMatrix::Identity();
            }

////            if (time ==  m_params.simulation_time - 1)
//                std::cout << time << "  " << cov << "\n";

            // Compute the feedback and the feedforward gain
            m_bp_vector[time].fb_k= -1 * Quu_reg.colPivHouseholderQr().solve(Qux_reg);
            m_bp_vector[time].ff_k = -1 * Quu_reg.colPivHouseholderQr().solve(Qu);

            //Approximate value functions
            V_x = Qx + (m_bp_vector[time].fb_k.transpose() * Quu * (m_bp_vector[time].ff_k));
            V_x += m_bp_vector[time].fb_k.transpose() * Qu + Qux.transpose() * m_bp_vector[time].ff_k;
            V_xx = Qxx + m_bp_vector[time].fb_k.transpose() * Quu * m_bp_vector[time].fb_k;
            V_xx += m_bp_vector[time].fb_k.transpose() * Qux + Qux.transpose() * m_bp_vector[time].fb_k;
            V_xx = 0.5 * (V_xx + V_xx.transpose());
            ++iter;
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
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, m_x_traj_new[time + 1], _m);
        }

        // Check if backtracking needs to continue
        expected_cost_red = compute_expected_cost(backtracker);
        new_total_cost = _cf.trajectory_running_cost(m_x_traj_new, m_u_traj_new);
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
            _cf.m_u_prev = m_u_traj.front();
            cost.emplace_back(_prev_total_cost);
        }
    }
    else{
        std::rotate(_covariance.begin(), _covariance.begin() + 1, _covariance.end());
        _covariance.back() = CtrlMatrix::Identity() * 0.15;

    }
    cached_control = m_u_traj.front();
    std::rotate(m_u_traj.begin(), m_u_traj.begin() + 1, m_u_traj.end());
    m_u_traj.back() = CtrlVector::Zero();

}
