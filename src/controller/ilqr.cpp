#include <iostream>
#include <numeric>
#include "mujoco.h"
#include "ilqr.h"
#include "../src/controller/simulation_params.h"

namespace
{
    template <typename T>
    void copy_data(const mjModel* model, const mjData *data, T *data_cp)
    {
        data_cp->time = data->time;
        mju_copy(data_cp->qpos, data->qpos, model->nq);
        mju_copy(data_cp->qvel, data->qvel, model->nv);
        mju_copy(data_cp->qacc, data->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data->xfrc_applied, 6*model->nbody);
        mju_copy(data_cp->ctrl, data->ctrl, model->nu);
    }


    template<typename T, int M, int N>
    void set_control_data(mjData* data, const Eigen::Matrix<T, M, N>& ctrl)
    {
        for(auto row = 0; row < ctrl.rows(); ++row)
        {
            data->qfrc_applied[row] = ctrl(row, 0);
        }
    }


    template<typename T, int M, int N>
    void fill_state_vector(mjData* data, Eigen::Matrix<T, M, N>& state)
    {
        state(0, 0) = data->qpos[0]; state(1, 0) = data->qpos[1];
        state(2, 0) = data->qvel[0]; state(3, 0) = data->qvel[1];
    }


    template<int rows, int cols>
    void clamp_control(Eigen::Matrix<mjtNum, rows, cols>& control, mjtNum max_bound, mjtNum min_bound)
    {
        for (auto row = 0; row < control.rows(); ++row)
        {
            control(row, 0) = std::clamp(control(row, 0), min_bound, max_bound);
        }
    }
}

using namespace SimulationParameters;

template<int state_size, int ctrl_size>
ILQR<state_size, ctrl_size>::ILQR(FiniteDifference<state_size, ctrl_size>& fd,
                                  CostFunction<state_size, ctrl_size>& cf,
                                  const mjModel * m,
                                  const int simulation_time) :
_fd(fd) ,_cf(cf), _m(m), _simulation_time(simulation_time)
{
    _d_cp = mj_makeData(m);

    _prev_total_cost = 0;
    _regularizer.setIdentity();

    _l.assign(_simulation_time + 1, 0);
    _l_x.assign(simulation_time + 1, Eigen::Matrix<double, state_size, 1>::Zero());
    _l_xx.assign(simulation_time + 1, Eigen::Matrix<double, state_size, state_size>::Zero());
    _l_u.assign(simulation_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());
    _l_ux.assign(simulation_time, Eigen::Matrix<double, ctrl_size, state_size>::Zero());
    _l_uu.assign(simulation_time, Eigen::Matrix<double, ctrl_size, ctrl_size>::Zero());

    _f_x.assign(simulation_time, Eigen::Matrix<double, state_size, state_size>::Zero());
    _f_u.assign(simulation_time, Eigen::Matrix<double, state_size, ctrl_size>::Zero());

    _fb_K.assign(_simulation_time, Eigen::Matrix<double, ctrl_size, state_size>::Zero());
    _ff_k.assign(_simulation_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());

    _x_traj_new.assign(_simulation_time + 1, Eigen::Matrix<double, state_size, 1>::Zero());
    _x_traj.assign(_simulation_time + 1, Eigen::Matrix<double, state_size, 1>::Zero());
    _u_traj.assign(_simulation_time, Eigen::Matrix<double, ctrl_size, 1>::Zero());

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
Eigen::Matrix<double, ctrl_size, ctrl_size>
ILQR<state_size, ctrl_size>::Q_uu(int time, Eigen::Matrix<double, state_size, state_size>& _v_xx)
{
    return _l_uu[time] + (_f_u[time].transpose() * (_v_xx + _regularizer)) * (_f_u[time]);
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_simulate(const mjData* d)
{
    if (recalculate)
    {
        copy_data(_m, d, _d_cp);
        _cf._d = _d_cp;
        for (auto time = 0; time < _simulation_time; ++time)
        {
            fill_state_vector(_d_cp, _x_traj[time]);
            set_control_data(_d_cp, _u_traj[time]);
            _l[time] = _cf.running_cost();
            _l_u[time] = (_cf.L_u());
            _l_x[time] = (_cf.L_x());
            _l_xx[time] = (_cf.L_xx());
            _l_ux[time] = (_cf.L_ux());
            _l_uu[time] = (_cf.L_uu());
            _fd.f_x_f_u(_d_cp);
            _f_x[time] = (_fd.f_x());
            _f_u[time] = (_fd.f_u());
            mj_step(_m, _d_cp);
        }
        _l.back()    = _cf.terminal_cost();
        _l_x.back()  = _cf.Lf_x();
        _l_xx.back() = _cf.Lf_xx();
        //TODO: Calculate the terminal costs
        copy_data(_m, d, _d_cp);
    }
    _prev_total_cost = std::accumulate(_l.begin(), _l.end(), 0.0);
}


// TODO: make data const if you can
template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::backward_pass()
{
    Eigen::Matrix<double, state_size, 1> V_x = _l_x.back();
    Eigen::Matrix<double, state_size, state_size> V_xx = _l_xx.back();

    for (auto time = _simulation_time - 1; time >= 0; --time)
    {
#if 0
        _ff_k[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_u(time, V_x));
        _fb_K[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_ux(time, V_xx));
#endif
        _ff_k[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_u(time, V_x));
        _fb_K[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_ux(time, V_xx));
        V_x   = Q_x(time, V_x) + (_fb_K[time].transpose() * Q_uu(time, V_xx)) * (_ff_k[time]);
        V_x  += _fb_K[time].transpose() * Q_u(time, V_x) + Q_ux(time, V_xx).transpose() * _ff_k[time];

        V_xx  = Q_xx(time, V_xx) + _fb_K[time].transpose() * Q_uu(time, V_xx) * _fb_K[time];
        V_xx += _fb_K[time].transpose() * Q_ux(time, V_xx) + Q_ux(time, V_xx).transpose() * _fb_K[time];

        V_xx  = 0.5 * (V_xx + V_xx.transpose());
    }
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_pass()
{

    for (const auto& backtracker : _backtrackers)
    {
        _x_traj_new.front() = _x_traj.front();
        std::fill(_u_traj.begin(), _u_traj.end(), Eigen::Matrix<double, ctrl_size, 1>::Zero());
        for (auto time = 0; time < _simulation_time; ++time) {
            _u_traj[time] = _u_traj[time] + (backtracker * _ff_k[time]) + _fb_K[time] * (_x_traj_new[time] - _x_traj[time]);
#if 0
            clamp_control(_u_traj[time], max_bound, min_bound);
            set_control_data(_d_cp, _u_traj[time]);
#endif
            set_control_data(_d_cp,_u_traj[time]);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _x_traj_new[time + 1]);
        }

        auto new_total_cost = _cf.trajectory_running_cost(_x_traj_new, _u_traj);

        if (new_total_cost < _prev_total_cost)
        {
            converged = (std::abs(_prev_total_cost - new_total_cost / _prev_total_cost) < 1e-6);
            _prev_total_cost = new_total_cost;
            recalculate = true;
            _delta = std::min(1.0, _delta) / _delta_init;
            _regularizer *= _delta;
            if (_regularizer.norm() < 1e-6)
                _regularizer.setZero();
            accepted = true;
            break;
        }
        if (not accepted)
        {
            _delta = std::max(1.0, _delta) * _delta_init;
            static const auto min = Matrix<double,state_size , state_size>::Identity() * 1e-6;
            // All elements are equal hence the (0, 0) comparison
            if ((_regularizer * _delta)(0, 0) > 1e-6)
            {
                _regularizer = _regularizer * _delta;
            }
            else{
                _regularizer = min;
            }
        }
    }
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::control(mjData* d)
{
    _delta = _delta_init;
    _regularizer.setIdentity();

    for(auto iteration = 0; iteration < 1; ++iteration)
    {
        forward_simulate(d);
        backward_pass();
        forward_pass();
        _cached_control = _u_traj.front();
        std::rotate(_u_traj.begin(), _u_traj.begin() + 1, _u_traj.end());
        _u_traj.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
        if (converged)
            break;
    }
}

template class ILQR<n_jpos + n_jvel, n_ctrl>;
