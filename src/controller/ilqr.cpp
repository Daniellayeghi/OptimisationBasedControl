#include <iostream>
#include <numeric>
#include "mujoco.h"
#include "ilqr.h"
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
    void set_control_data(mjData* data, const Eigen::Matrix<T, M, N> ctrl)
    {
        for(auto row = 0; row < ctrl.rows(); ++row)
        {
            data->qfrc_applied[row] = ctrl(row, 0);
        }
    }

    template<typename T, int M, int N>
    void fill_state_vector(mjData* data, Eigen::Matrix<T, M, N> state)
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


ILQR::ILQR(FiniteDifference& fd, CostFunction& cf, const mjModel * m, int simulation_time) :
_fd(fd) ,_cf(cf), _m(m), _simulation_time(simulation_time)
{
    _d_cp = mj_makeData(m);

    _F.reserve(_simulation_time);
    _F_x.reserve(_simulation_time);
    _F_u.reserve(_simulation_time);

    _L_u.reserve(_simulation_time);
    _L_ux.reserve(_simulation_time);
    _L_uu.reserve(_simulation_time);

    _L.reserve(_simulation_time + 1);
    _L_x.reserve(_simulation_time + 1);
    _L_xx.reserve(_simulation_time + 1);

    _ff_K.reserve(_simulation_time);
    _fb_k.reserve(_simulation_time);

    _x_traj_new.reserve(_simulation_time + 1);
    _x_traj.reserve(_simulation_time + 1);
    _u_traj.reserve(_simulation_time);

    for(auto i =0; i < _simulation_time; ++i)
    {
        _L.emplace_back(0);
        _F_x.emplace_back(Eigen::Matrix<double, 4, 4>::Zero());
        _F_u.emplace_back(Eigen::Matrix<double, 4, 2>::Zero());
        _L_x.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
        _L_u.emplace_back(Eigen::Matrix<double, 2, 1>::Zero());
        _L_xx.emplace_back(Eigen::Matrix<double, 4, 4>::Zero());
        _L_ux.emplace_back(Eigen::Matrix<double, 2, 4>::Zero());
        _L_uu.emplace_back(Eigen::Matrix<double, 2, 2>::Zero());
        _ff_K.emplace_back(Eigen::Matrix<double, 2, 4>::Zero());
        _fb_k.emplace_back(Eigen::Matrix<double, 2, 1>::Zero());
        _u_traj.emplace_back(Eigen::Matrix<double, 2, 1>::Zero());
        _x_traj.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
        _x_traj_new.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
    };
}


ILQR::~ILQR()
{
    mj_deleteData(_d_cp);
}


Eigen::Matrix<mjtNum, 4, 1> ILQR::Q_x(int time, InternalTypes::Mat4x1& _V_x)
{
    return _L_x[time] + _F_x[time].transpose() * _V_x ;
}


Eigen::Matrix<mjtNum, 2, 1> ILQR::Q_u(int time,  InternalTypes::Mat4x1& _V_x)
{
    return _L_u[time] + _F_u[time].transpose() * _V_x ;
}


Eigen::Matrix<mjtNum, 4, 4> ILQR::Q_xx(int time, InternalTypes::Mat4x4& _V_xx)
{
    return _L_xx[time] + _F_x[time].transpose() * _V_xx * _F_x[time];
}


Eigen::Matrix<mjtNum, 2, 4> ILQR::Q_ux(int time, InternalTypes::Mat4x4& _V_xx)
{
    return _L_ux[time] + _F_u[time].transpose() * (_V_xx) * _F_x[time];
}


Eigen::Matrix<mjtNum, 2, 2> ILQR::Q_uu(int time, InternalTypes::Mat4x4& _V_xx)
{
    return _L_uu[time] + _F_u[time].transpose() * (_V_xx) * _F_u[time];
}


void ILQR::forward_simulate(const mjData* d)
{
    if (recalculate)
    {
        copy_data(_m, d, _d_cp);
        _cf._d = _d_cp;
        for (auto time = 0; time < _simulation_time; ++time)
        {
            fill_state_vector(_d_cp, _x_traj[time]);
            _fd.f_x_f_u(_d_cp);
            _F_x[time] = (_fd.f_x(_d_cp));
            _F_u[time] = (_fd.f_u(_d_cp));
            _L[time] = _cf.running_cost();
            _L_u[time] = (_cf.L_u());
            _L_x[time] = (_cf.L_x());
            _L_xx[time] = (_cf.L_xx());
            _L_ux[time] = (_cf.L_ux());
            _L_uu[time] = (_cf.L_uu());
            mj_step(_m, _d_cp);
        }
        _L.back()    = _cf.terminal_cost();
        _L_x.back()  = _cf.Lf_x();
        _L_xx.back() = _cf.Lf_xx();
        //TODO: Calculate the terminal costs
        copy_data(_m, d, _d_cp);
        recalculate = false;
    }
}


// TODO: make data const if you can
void ILQR::backward_pass(mjData* d)
{
    InternalTypes::Mat4x1 V_x = _L_x.back();
    InternalTypes::Mat4x4 V_xx = _L_xx.back();

    for (auto time = _simulation_time - 1; time >= 0; --time)
    {
        _fb_k[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_u(time, V_x));
        _ff_K[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_ux(time, V_xx));
        V_x = Q_x(time, V_x) + _ff_K[time].transpose() * Q_uu(time, V_xx) * _fb_k[time];
        V_x += _ff_K[time].transpose() * Q_u(time, V_x) + Q_ux(time, V_xx).transpose() * _fb_k[time];
        V_xx = Q_xx(time, V_xx) + _ff_K[time].transpose() * Q_uu(time,V_xx) * _ff_K[time];
        V_xx += _ff_K[time].transpose() * Q_ux(time, V_xx) + Q_ux(time, V_xx).transpose() * _ff_K[time];
        V_xx = 0.5 * (V_xx + V_xx.transpose());
    }
}


void ILQR::forward_pass()
{

    _x_traj_new.front() = _x_traj.front();
    for (auto time = 0; time < _simulation_time; ++time) {
        _u_traj[time] = _u_traj[time] + _fb_k[time] + _ff_K[time] * (_x_traj_new[time] - _x_traj[time]);
        clamp_control(_u_traj[time], max_bound, min_bound);
        set_control_data(_d_cp, _u_traj[time]);
        mj_step(_m, _d_cp);
        fill_state_vector(_d_cp, _x_traj_new[time + 1]);
    }

    auto new_total_cost = _cf.trajectory_running_cost(_x_traj_new, _u_traj);
    auto prev_total_cost = std::accumulate(_L.begin(), _L.end(), 0.0);

    if (new_total_cost < prev_total_cost)
    {
        converged = (std::abs(prev_total_cost - new_total_cost / prev_total_cost) < 1e-6);
        _x_traj = _x_traj_new;
        recalculate = true;
    }

}


void ILQR::control(mjData* d)
{
    for(auto iteration = 0; iteration < 5; ++iteration)
    {
        forward_simulate(d);
        backward_pass(d);
        forward_pass();
        _cached_control = _u_traj.front();
        std::rotate(_u_traj.begin(), _u_traj.begin() + 1, _u_traj.end());
        _u_traj.back() = InternalTypes::Mat2x1::Zero();
        if (converged)
            break;
    }
}


Eigen::Ref<InternalTypes::Mat2x1> ILQR::get_control()
{
    return _cached_control;
}

