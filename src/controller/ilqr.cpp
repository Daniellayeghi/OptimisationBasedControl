#include<iostream>
#include <numeric>
#include "mujoco.h"
#include "ilqr.h"
#include "../parameters/simulation_params.h"
#include "../../src/utilities/mujoco_utils.h"

using namespace MujocoUtils;
using namespace SimulationParameters;


//namespace
//{
//    template<int state_size, int ctrl_size>
//    Eigen::Matrix<double, state_size+ctrl_size, state_size+ctrl_size>
//    near_psd(const Eigen::Matrix<double, state_size+ctrl_size, state_size+ctrl_size>& matrix)
//    {
//        auto sym_mat = (matrix + matrix.transpose().eval())/2;
//        Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU|Eigen::ComputeFullV);
//        auto sym_polar_factor = svd.matrixV() * svd.singularValues() * svd.matrixV();
//        auto matrix_hat = (sym_mat + sym_polar_factor)/2;
//        matrix_hat = (matrix_hat + matrix_hat.transpose())/2;
//        auto k = 0; auto p = true;
//
//        while(p)
//        {
//            Eigen::LLT<Eigen::MatrixXd> lltOfA(matrix_hat);
//            p = lltOfA.info() == Eigen::NumericalIssue;
//            ++k;
//            if (p)
//            {
//                Eigen::EigenSolver<Eigen::MatrixXf> es;
//                es.compute(matrix_hat);
//                auto* eigen_val = es.eigenvalues().data();
//                std::complex min(0, 0); min.real(std::numeric_limits<float>::min());
//                for (int elem = 0; elem < es.eigenvalues().size(); ++elem)
//                {
//                    auto min =
//                }
//
//            }
//
//        }
//    }
//}


template<int state_size, int ctrl_size>
ILQR<state_size, ctrl_size>::ILQR(FiniteDifference<state_size, ctrl_size>& fd,
                                  CostFunction<state_size, ctrl_size>& cf,
                                  const mjModel * m,
                                  const int simulation_time,
                                  const int iteration,
                                  const mjData* d,
                                  const std::vector<ILQR<state_size, ctrl_size>::ctrl_vec>* init_u) :
_fd(fd) ,_cf(cf), _m(m), _simulation_time(simulation_time), _iteration(iteration)
{
    _d_cp = mj_makeData(m);
    _prev_total_cost = 0;
    _regularizer.setIdentity();

    exp_cost_reduction.assign(_simulation_time, 0.0);
    _l.assign(_simulation_time + 1, 0);
    _l_x.assign(simulation_time + 1, state_vec::Zero());
    _l_xx.assign(simulation_time + 1, state_mat::Zero());
    _l_u.assign(simulation_time, ctrl_vec::Zero());
    _l_ux.assign(simulation_time, ctrl_state_mat::Zero());
    _l_uu.assign(simulation_time, ctrl_mat::Zero());

    _f_x.assign(simulation_time, state_mat::Zero());
    _f_u.assign(simulation_time, state_ctrl_mat::Zero());

    _fb_K.assign(_simulation_time, ctrl_state_mat ::Zero());
    _ff_k.assign(_simulation_time, ctrl_vec::Zero());

    _x_traj_new.assign(_simulation_time + 1, state_vec::Zero());
    _x_traj.assign(_simulation_time + 1, state_vec::Zero());
    _u_traj_new.assign(_simulation_time, ctrl_vec::Zero());
    _u_traj_cp.assign(_simulation_time,ctrl_vec::Zero());
    _covariance.assign(_simulation_time, ctrl_mat::Zero());

    if(init_u == nullptr)
    {
        _u_traj.assign(_simulation_time,ctrl_vec::Random() * 0);

    }else
    {
        _u_traj = *init_u;
    }

    copy_data(m, d, _d_cp);
    fill_state_vector(d, _x_traj.front());
    for (int time = 0; time < simulation_time; ++time)
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
                       2.24320079e-03, 4.43805318e-04}};
}


template<int state_size, int ctrl_size>
ILQR<state_size, ctrl_size>::~ILQR()
{
    mj_deleteData(_d_cp);
}


template<int state_size, int ctrl_size>
Eigen::Matrix<double, state_size, 1>
ILQR<state_size, ctrl_size>::Q_x(int time, Eigen::Matrix<double, state_size, 1>& _v_x)
{
    return _l_x[time] + _f_x[time].transpose() * _v_x ;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<double, ctrl_size, 1>
ILQR<state_size, ctrl_size>::Q_u(int time,  Eigen::Matrix<double, state_size, 1>& _v_x)
{
    return _l_u[time] + _f_u[time].transpose() * _v_x ;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, state_size>
ILQR<state_size, ctrl_size>::Q_xx(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx)
{
    return _l_xx[time] + (_f_x[time].transpose() * _v_xx) * _f_x[time];
}


template<int state_size, int ctrl_size>
Eigen::Matrix<double, ctrl_size, state_size>
ILQR<state_size, ctrl_size>::Q_ux(int time,Eigen::Matrix<double, state_size, state_size>& _v_xx)
{
    return _l_ux[time] + (_f_u[time].transpose() * (_v_xx + _regularizer)) * _f_x[time];
}

template<int state_size, int ctrl_size>
Eigen::Matrix<double, state_size, ctrl_size>
ILQR<state_size, ctrl_size>::Q_xu(int time,Eigen::Matrix<double, state_size, state_size>& _v_xx)
{
    return _f_x[time].transpose() * (_v_xx + _regularizer) * _f_u[time];
}


template<int state_size, int ctrl_size>
Eigen::Matrix<double, ctrl_size, ctrl_size>
ILQR<state_size, ctrl_size>::Q_uu(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx)
{
    return _l_uu[time] + (_f_u[time].transpose() * (_v_xx+_regularizer)) * (_f_u[time]);
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_simulate(const mjData* d)
{
    if (recalculate)
    {
        copy_data(_m, d, _d_cp);
        for (auto time = 0; time < _simulation_time; ++time)
        {
            set_control_data(_d_cp, _u_traj[time]);
            _l[time] = _cf.running_cost(_d_cp);
            _l_u[time] = (_cf.L_u(_d_cp));
            _l_x[time] = (_cf.L_x(_d_cp));
            _l_xx[time] = (_cf.L_xx(_d_cp));
            _l_ux[time] = (_cf.L_ux(_d_cp));
            _l_uu[time] = (_cf.L_uu(_d_cp));
            _fd.f_x_f_u(_d_cp);
            _f_x[time] = (_fd.f_x());
            _f_u[time] = (_fd.f_u());
            mj_step(_m, _d_cp);
        }
        _l.back()    = _cf.terminal_cost(_d_cp);
        _l_x.back()  = _cf.Lf_x(_d_cp);
        _l_xx.back() = _cf.Lf_xx();
        copy_data(_m, d, _d_cp);
        _prev_total_cost = std::accumulate(_l.begin(), _l.end(), 0.0);
        recalculate = false;
    }
}


// TODO: make data const if you can
template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::backward_pass()
{
    Eigen::Matrix<double, state_size, 1> V_x = _l_x.back();
    Eigen::Matrix<double, state_size, state_size> V_xx = _l_xx.back();
    static Eigen::Matrix<double, state_size+ctrl_size, state_size+ctrl_size> hessian;
//    Eigen::JacobiSVD<Eigen::MatrixXd> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    for (auto time = _simulation_time - 1; time >= 0; --time)
    {
        const auto Qx = Q_x(time, V_x);    const auto Qu  = Q_u(time, V_x); const auto Qxu = Q_xu(time, V_xx);
        const auto Quu = Q_uu(time, V_xx); const auto Qux = Q_ux(time, V_xx); const auto Qxx = Q_xx(time, V_xx);
        const ctrl_mat cov = 1/1*(Quu - Qux*Qxx.inverse()*Qxu).inverse();
        hessian << Qxx, Qxu, Qux, Quu;
        Eigen::LLT<Eigen::MatrixXd> lltOfA(hessian);
        auto p = lltOfA.info() == Eigen::NumericalIssue;
        std::cout << "Is PSD: " << p << "\n";
        if (std::any_of(cov.data(), cov.data() + cov.size(), [](double val){return not std::isnan(val);})) {
            _covariance[time] = cov;
        }
        else {
            _covariance[time] = ctrl_mat::Identity();
        }
//        JacobiSVD<MatrixXd> svd(Quu);
//        double cond = svd.singularValues()(0)/svd.singularValues()(svd.singularValues().size()-1);
//        exp_cost_reduction[time] = (Q_u(time, V_x).transpose() * _ff_k[time]) - (0.5 * _ff_k[time].transpose() * (Q_uu(time, V_xx) * _ff_k[time]))(0, 0);
//        exp_cost_reduction[time] = cond;
//        _fb_K[time] = -1 * Quu.bdcSvd(ComputeFullU | ComputeFullV).solve(Qux);
//        _ff_k[time] = -1 * Quu.bdcSvd(ComputeFullU | ComputeFullV).solve(Qu);
        _fb_K[time] = -1 * Quu.colPivHouseholderQr().solve(Qux);
        _ff_k[time] = -1 * Quu.colPivHouseholderQr().solve(Qu);

        V_x   = Qx + (_fb_K[time].transpose() * Quu * (_ff_k[time]));
        V_x  += _fb_K[time].transpose() * Qu + Qux.transpose() * _ff_k[time];
        V_xx  = Qxx + _fb_K[time].transpose() * Quu * _fb_K[time];
        V_xx += _fb_K[time].transpose() * Qux + Qux.transpose() * _fb_K[time];
        V_xx  = 0.5 * (V_xx + V_xx.transpose());
    }
}


template<int state_size, int ctrl_size>
auto ILQR<state_size, ctrl_size>::update_regularizer()
{
    auto break_condition = false;
    auto new_total_cost = _cf.trajectory_running_cost(_x_traj_new, _u_traj_new);
    if (new_total_cost < _prev_total_cost or new_total_cost < 1e-8)
    {
        converged = (std::abs(_prev_total_cost - new_total_cost / _prev_total_cost) < 1e-6);
        _prev_total_cost = new_total_cost;
        recalculate = true;
        _delta = std::min(1.0, _delta) / _delta_init;
        _regularizer *= _delta;
        if (_regularizer.norm() < 1e-6)
            _regularizer.setIdentity() * 1e-6;

        accepted = true;
        _x_traj = _x_traj_new;
        _u_traj = _u_traj_new;
        break_condition = true;
    }

    if (not accepted) {
        _delta = std::max(1.0, _delta) * _delta_init;
        static const auto min = ILQR::state_mat::Identity() * 1e-6;
        // All elements are equal hence the (0, 0) comparison
        if ((_regularizer * _delta)(0, 0) > 1e-6) {
            _regularizer = _regularizer * _delta;
        } else {
            _regularizer = min;
        }
        if (_regularizer(0, 0) > 1e10) {
//                std::cout << "Exceed" "\n";
            break_condition = true;
        }
    }
    return break_condition;
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_pass(const mjData* d)
{
    //TODO Regularize the Quu inversion instead
    for (const auto &backtracker : _backtrackers)
    {
        std::fill(_u_traj_new.begin(), _u_traj_new.end(), ctrl_vec::Zero());

        copy_data(_m, d, _d_cp);
        _x_traj_new.front() = _x_traj.front();
        for (auto time = 0; time < _simulation_time; ++time)
        {
            _u_traj_new[time] =  _u_traj[time] + (_ff_k[time] * backtracker) + _fb_K[time] * (_x_traj_new[time] - _x_traj[time]);
            clamp_control(_u_traj_new[time], _m->actuator_ctrlrange);
            set_control_data(_d_cp, _u_traj_new[time]);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _x_traj_new[time + 1]);
        }
        if(update_regularizer())
            break;
    }
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::control(const mjData* d)
{
    _delta = _delta_init; recalculate = true; converged = false;
    _regularizer.setIdentity();
    for(auto iteration = 0; iteration < _iteration; ++iteration)
    {
        recalculate = true; converged = false; accepted = false; _delta = _delta_init;
        _regularizer.setIdentity();
        fill_state_vector(d, _x_traj.front());
        forward_simulate(d);
        backward_pass();
        forward_pass(d);
        if (converged)
            break;
    }
    _cached_control = _u_traj.front();
    std::copy(_u_traj.begin(), _u_traj.end(), _u_traj_cp.begin());
    std::rotate(_u_traj.begin(), _u_traj.begin() + 1, _u_traj.end());
    _u_traj.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
    cost.emplace_back(_prev_total_cost);
}


template class ILQR<n_jpos + n_jvel, n_ctrl>;
