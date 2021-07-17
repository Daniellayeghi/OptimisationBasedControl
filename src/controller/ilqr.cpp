#include<iostream>
#include <numeric>
#include "mujoco.h"
#include "ilqr.h"
#include "../parameters/simulation_params.h"
#include "../../src/utilities/mujoco_utils.h"

using namespace MujocoUtils;
using namespace SimulationParameters;

template<int state_size, int ctrl_size>
ILQR<state_size, ctrl_size>::ILQR(FiniteDifference<state_size, ctrl_size>& fd,
                                  CostFunction<state_size, ctrl_size>& cf,
                                  ILQRParams& params,
                                  const mjModel * m,
                                  const mjData* d,
                                  const std::vector<CtrlVector>* init_u) :
_fd(fd) ,_cf(cf), _m(m), m_params(params)
{
    _d_cp = mj_makeData(m);
    _prev_total_cost = 0;
    _regularizer.setIdentity();

    exp_cost_reduction.assign(m_params.simulation_time, 0.0);


    m_d_vector.assign(m_params.simulation_time + 1,{0, StateVector::Zero(), StateMatrix::Zero(),
                                            CtrlVector::Zero(), CtrlMatrix::Zero(), CtrlStateMatrix::Zero(),
                                            StateMatrix::Zero(), StateCtrlMatrix::Zero()});

    m_bp_vector.assign(m_params.simulation_time + 1, {CtrlVector::Zero(), CtrlStateMatrix::Zero()});

    _x_traj_new.assign(m_params.simulation_time + 1, StateVector::Zero());
    _x_traj.assign(m_params.simulation_time + 1, StateVector::Zero());
    _u_traj_new.assign(m_params.simulation_time, CtrlVector::Zero());
    _u_traj_cp.assign(m_params.simulation_time,CtrlVector::Zero());
    _covariance.assign(m_params.simulation_time, CtrlMatrix::Identity());
    m_Qu_traj.assign(m_params.simulation_time, CtrlVector::Zero());
    m_Quu_traj.assign(m_params.simulation_time, CtrlMatrix::Zero());


    if(init_u == nullptr)
    {
        _u_traj.assign(m_params.simulation_time, CtrlVector::Random() * 0);

    }else
    {
        _u_traj = *init_u;
    }

    copy_data(m, d, _d_cp);
    fill_state_vector(d, _x_traj.front());
    for (int time = 0; time < m_params.simulation_time; ++time)
    {
        set_control_data(_d_cp, _u_traj[time]);
        mj_step(m, _d_cp);
        fill_state_vector(_d_cp, _x_traj[time+1]);
    }
    copy_data(m, d, _d_cp);

    _backtrackers =  {{1.00000000e+00, 9.09090909e-01,
                       6.83013455e-01, 4.24097618e-01,
                       2.17629136e-01, 9.22959982e-02,
                       3.23491843e-02, 9.37040641e-03,
                       2.24320079e-03, 4.43805318e-04, 0.00000001}};
}


template<int state_size, int ctrl_size>
ILQR<state_size, ctrl_size>::~ILQR()
{
    mj_deleteData(_d_cp);
}


template<int state_size, int ctrl_size>
StateVector ILQR<state_size, ctrl_size>::Q_x(int time, StateVector& _v_x)
{
    return m_d_vector[time].lx + m_d_vector[time].fx.transpose().eval() * _v_x ;
}


template<int state_size, int ctrl_size>
CtrlVector ILQR<state_size, ctrl_size>::Q_u(int time,  StateVector& _v_x)
{
    return m_d_vector[time].lu + m_d_vector[time].fu.transpose().eval() * _v_x ;
}


template<int state_size, int ctrl_size>
StateMatrix ILQR<state_size, ctrl_size>::Q_xx(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lxx + (m_d_vector[time].fx.transpose().eval() * _v_xx) * m_d_vector[time].fx;
}


template<int state_size, int ctrl_size>
CtrlStateMatrix ILQR<state_size, ctrl_size>::Q_ux(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lux + m_d_vector[time].fu.transpose().eval() * (_v_xx) * m_d_vector[time].fx;
}

template<int state_size, int ctrl_size>
StateCtrlMatrix ILQR<state_size, ctrl_size>::Q_xu(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].fx.transpose().eval() * (_v_xx) * m_d_vector[time].fu;
}


template<int state_size, int ctrl_size>
CtrlMatrix ILQR<state_size, ctrl_size>::Q_uu(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].luu + (m_d_vector[time].fu.transpose().eval() * (_v_xx)) * (m_d_vector[time].fu);
}


template<int state_size, int ctrl_size>
StateMatrix ILQR<state_size, ctrl_size>::Q_xx_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lxx + (m_d_vector[time].fx.transpose().eval() * _v_xx + _regularizer) * m_d_vector[time].fx;
}


template<int state_size, int ctrl_size>
CtrlStateMatrix ILQR<state_size, ctrl_size>::Q_ux_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].lux + (m_d_vector[time].fu.transpose().eval() * (_v_xx + _regularizer)) * m_d_vector[time].fx;
}

template<int state_size, int ctrl_size>
StateCtrlMatrix ILQR<state_size, ctrl_size>::Q_xu_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].fx.transpose().eval() * (_v_xx + _regularizer) * m_d_vector[time].fu;
}


template<int state_size, int ctrl_size>
CtrlMatrix ILQR<state_size, ctrl_size>::Q_uu_reg(int time, StateMatrix& _v_xx)
{
    return m_d_vector[time].luu + (m_d_vector[time].fu.transpose().eval() * (_v_xx+_regularizer)) * (m_d_vector[time].fu);
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_simulate(const mjData* d)
{
    _prev_total_cost = 0;
    copy_data(_m, d, _d_cp);
    for (auto time = 0; time < m_params.simulation_time; ++time)
    {
        set_control_data(_d_cp, _u_traj[time]);
        _fd.f_x_f_u(_d_cp);
        _prev_total_cost += _cf.running_cost(_d_cp);
        m_d_vector[time].l = _cf.running_cost(_d_cp);
        m_d_vector[time].lx = _cf.L_x(_d_cp);
        m_d_vector[time].lxx = _cf.L_xx(_d_cp);
        m_d_vector[time].lu = _cf.L_u(_d_cp);
        m_d_vector[time].luu = _cf.L_uu(_d_cp);
        m_d_vector[time].lux = _cf.L_ux(_d_cp);
        m_d_vector[time].fx = _fd.f_x();
        m_d_vector[time].fu = _fd.f_u();
        mj_step(_m, _d_cp);
    }

    _prev_total_cost += _cf.terminal_cost(_d_cp);
    m_d_vector.back().l = _cf.terminal_cost(_d_cp);
    m_d_vector.back().lx = _cf.Lf_x(_d_cp);
    m_d_vector.back().lxx = _cf.Lf_xx();
    copy_data(_m, d, _d_cp);
}


// TODO: make data const if you can
template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::backward_pass()
{
    auto non_pd_path = false;
    Eigen::Matrix<double, state_size, 1> V_x = m_d_vector.back().lx;
    Eigen::Matrix<double, state_size, state_size> V_xx = m_d_vector.back().lxx;
    Eigen::Matrix<double, state_size+ctrl_size, state_size+ctrl_size> hessian;

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
                    Eigen::Matrix<double, state_size+ctrl_size, state_size+ctrl_size>::Identity()
                    );

            const CtrlMatrix cov = hessian_inverse.block(
                    state_size, state_size, ctrl_size, ctrl_size
                    );
//            const CtrlMatrix cov = 1 / 1 * (Quu_reg - Qux_reg * (Qxx_reg).inverse() * Qxu_reg).inverse();

            Eigen::LLT<Eigen::MatrixXd> lltOfA(Quu_reg);
            auto p = lltOfA.info() == Eigen::NumericalIssue;
            if (p) {non_pd_path = true; update_regularizer(true); break;}

            if (std::any_of(cov.data(), cov.data() + cov.size(), [](double val) {return not std::isnan(val);}))
            {
                _covariance[time] = cov;
            } else {
                _covariance[time] = CtrlMatrix::Identity();
            }

            // Compute the feedback and the feedforward gain
            m_bp_vector[time].fb_k = -1 * Quu_reg.colPivHouseholderQr().solve(Qux_reg);
            m_bp_vector[time].ff_k = -1 * Quu_reg.colPivHouseholderQr().solve(Qu);

            //Approximate value functions
            V_x = Qx + (m_bp_vector[time].fb_k.transpose().eval() * Quu * (m_bp_vector[time].ff_k));
            V_x += m_bp_vector[time].fb_k.transpose().eval() * Qu + Qux.transpose().eval() * m_bp_vector[time].ff_k;
            V_xx = Qxx + m_bp_vector[time].fb_k.transpose().eval() * Quu * m_bp_vector[time].fb_k;
            V_xx += m_bp_vector[time].fb_k.transpose().eval() * Qux + Qux.transpose().eval() * m_bp_vector[time].fb_k;
            V_xx = 0.5 * (V_xx + V_xx.transpose().eval());
        }
    }while(non_pd_path);
    update_regularizer(false);
//    temporal_average_covariance();
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::temporal_average_covariance()
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


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::update_regularizer(const bool increase)
{

    if(increase)
    {
        m_params.delta = std::max(m_params.delta_init, m_params.delta * m_params.delta_init);
        // All elements are equal hence the (0, 0) comparison
        if ((_regularizer * m_params.delta)(0, 0) > m_params.min_reg) {
            _regularizer = _regularizer * m_params.delta;
        } else {
            _regularizer = StateMatrix::Identity() * m_params.min_reg;
        }
    }else
    {
        m_params.delta = std::min(1.0/m_params.delta_init, m_params.delta/m_params.delta_init);
        if ((_regularizer * m_params.delta)(0, 0) > m_params.min_reg) {
            _regularizer = _regularizer * m_params.delta;
        } else {
            _regularizer = StateMatrix::Zero();
        }
    }
}


template<int state_size, int ctrl_size>
double ILQR<state_size, ctrl_size>::compute_expected_cost(const double backtracker)
{
    double estimate_1st = 0 , estimate_2nd = 0;
    for (auto time = 0; time < m_params.simulation_time; ++time)
    {
        estimate_1st += (-backtracker*(m_bp_vector[time].ff_k.transpose().eval() * m_Qu_traj[time])(0.0));
        estimate_2nd += (-backtracker*backtracker/2*(m_bp_vector[time].ff_k.transpose().eval() * m_Quu_traj[time] * m_bp_vector[time].ff_k)(0.0));
    }
    return estimate_1st + estimate_2nd;
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_pass(const mjData* d)
{
    static std::string status = "N";
    auto expected_cost_red = 0.0; auto new_total_cost = 0.0; auto cost_red_ratio = 0.0;

//    std::vector<std::vector<CtrlVector>> u_nominees; u_nominees.reserve(10);
//    u_nominees.assign(10, std::vector<CtrlVector>(m_params.simulation_time, CtrlVector::Zero()));
//    std::array<double, 10> costreds; costreds.fill(-1);
    //TODO Regularize the Quu inversion instead
    for (const auto &backtracker : _backtrackers)
    {
        std::fill(_u_traj_new.begin(), _u_traj_new.end(), CtrlVector::Zero());

        copy_data(_m, d, _d_cp);
        _x_traj_new.front() = _x_traj.front();
        for (auto time = 0; time < m_params.simulation_time; ++time)
        {
            _u_traj_new[time] =  _u_traj[time] + (m_bp_vector[time].ff_k * backtracker) + m_bp_vector[time].fb_k * (_x_traj_new[time] - _x_traj[time]);
            clamp_control(_u_traj_new[time], _m->actuator_ctrlrange);
            set_control_data(_d_cp, _u_traj_new[time]);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _x_traj_new[time + 1]);
        }

        // Check if backtracking needs to continue
        expected_cost_red = compute_expected_cost(backtracker);
        new_total_cost = _cf.trajectory_running_cost(_x_traj_new, _u_traj_new);
        cost_red_ratio = (_prev_total_cost - new_total_cost)/expected_cost_red;
//        costreds[iteration] = cost_red_ratio;
//        u_nominees[iteration] = _u_traj_new;

        // NOTE: Not doing this and updating regardless of the cost can lead to better performance!
        if(cost_red_ratio >= m_params.min_cost_red) {
            status = "Y";
            _u_traj = _u_traj_new;
            _x_traj = _x_traj_new;
            printf("Cost = %f, Cost Diff = %f, Expected Diff = %f, Lambda = %f, Update = %s, last_position = %f\n",
                   _prev_total_cost, _prev_total_cost - new_total_cost, expected_cost_red, _regularizer(0.0), status.c_str(), _x_traj_new.front()(0, 0));
            break;
        }
        else if (cost_red_ratio < 0){
            status = "N";
        }
    }

//    const auto element = std::max_element(costreds.begin(), costreds.end()) - costreds.begin();
//    if(costreds[element] > 0)
//        _u_traj = u_nominees[element];
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::control(const mjData* d)
{
    for(auto iteration = 0; iteration < m_params.iteration; ++iteration)
    {
        m_params.min_cost_red = 0;
        m_params.delta = m_params.delta_init;
        _regularizer.setIdentity();
        fill_state_vector(d, _x_traj.front());
        forward_simulate(d);
        backward_pass();
        forward_pass(d);
    }
    _cached_control = _u_traj.front();
    std::copy(_u_traj.begin(), _u_traj.end(), _u_traj_cp.begin());
    std::rotate(_u_traj.begin(), _u_traj.begin() + 1, _u_traj.end());
    _u_traj.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
    cost.emplace_back(_prev_total_cost);
}


template class ILQR<n_jpos + n_jvel, n_ctrl>;
