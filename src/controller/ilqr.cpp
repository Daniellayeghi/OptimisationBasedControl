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


ILQR::ILQR(FiniteDifference& fd, CostFunction& cf, const mjModel * m, int simulation_time) :
_fd(fd) ,_cf(cf), _m(m), _simulation_time(simulation_time)
{
    _d_cp = mj_makeData(m);

    _f.reserve(_simulation_time);
    _f_x.reserve(_simulation_time);
    _f_u.reserve(_simulation_time);

    _l_u.reserve(_simulation_time);
    _l_ux.reserve(_simulation_time);
    _l_uu.reserve(_simulation_time);

    _l.reserve(_simulation_time + 1);
    _l_x.reserve(_simulation_time + 1);
    _l_xx.reserve(_simulation_time + 1);

    _fb_K.reserve(_simulation_time);
    _ff_k.reserve(_simulation_time);

    _x_traj_new.reserve(_simulation_time + 1);
    _x_traj.reserve(_simulation_time + 1);
    _u_traj.reserve(_simulation_time);

    for(auto i =0; i < _simulation_time; ++i)
    {
        _l.emplace_back(0);
        _f_x.emplace_back(Eigen::Matrix<double, 4, 4>::Zero());
        _f_u.emplace_back(Eigen::Matrix<double, 4, 2>::Zero());
        _l_x.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
        _l_u.emplace_back(Eigen::Matrix<double, 2, 1>::Zero());
        _l_xx.emplace_back(Eigen::Matrix<double, 4, 4>::Zero());
        _l_ux.emplace_back(Eigen::Matrix<double, 2, 4>::Zero());
        _l_uu.emplace_back(Eigen::Matrix<double, 2, 2>::Zero());
        _fb_K.emplace_back(Eigen::Matrix<double, 1, 4>::Zero());
        _x_traj.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
        _x_traj_new.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
        _ff_k.emplace_back(0);
        _u_traj.emplace_back(0);
    };
}


ILQR::~ILQR()
{
    mj_deleteData(_d_cp);
}


Eigen::Matrix<mjtNum, 4, 1> ILQR::Q_x(int time, InternalTypes::Mat4x1& _v_x)
{
    return _l_x[time] + _f_x[time].transpose() * _v_x ;
}


double ILQR::Q_u(int time,  InternalTypes::Mat4x1& _v_x)
{
    return _l_u[time](1, 0) + _f_u[time].block<4, 1>(0, 1).transpose() * _v_x ;
}


Eigen::Matrix<mjtNum, 4, 4> ILQR::Q_xx(int time, InternalTypes::Mat4x4& _v_xx)
{
    return _l_xx[time] + _f_x[time].transpose() * _v_xx * _f_x[time];
}


Eigen::Matrix<mjtNum, 1, 4> ILQR::Q_ux(int time, InternalTypes::Mat4x4& _v_xx)
{
    return _l_ux[time].block<1, 4>(1, 0) +
            _f_u[time].block<4, 1>(0, 1).transpose() * (_v_xx) * _f_x[time];
}


double ILQR::Q_uu(int time, InternalTypes::Mat4x4& _v_xx)
{
    return _l_uu[time](1, 1) + _f_u[time].block<4, 1>(0, 1).transpose() * (_v_xx) *
                                        _f_u[time].block<4, 1>(0, 1);
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
            _f_x[time] = (_fd.f_x(_d_cp));
            _f_u[time] = (_fd.f_u(_d_cp));
            _l[time] = _cf.running_cost();
            _l_u[time] = (_cf.L_u());
            _l_x[time] = (_cf.L_x());
            _l_xx[time] = (_cf.L_xx());
            _l_ux[time] = (_cf.L_ux());
            _l_uu[time] = (_cf.L_uu());
            mj_step(_m, _d_cp);
        }
        _l.back()    = _cf.terminal_cost();
        _l_x.back()  = _cf.Lf_x();
        _l_xx.back() = _cf.Lf_xx();
        //TODO: Calculate the terminal costs
        copy_data(_m, d, _d_cp);
        recalculate = false;
    }
}


// TODO: make data const if you can
void ILQR::backward_pass()
{
    InternalTypes::Mat4x1 V_x = _l_x.back();
    InternalTypes::Mat4x4 V_xx = _l_xx.back();

    for (auto time = _simulation_time - 1; time >= 0; --time)
    {
//        _ff_k[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_u(time, V_x));
//        _fb_K[time] = -1 * Q_uu(time, V_xx).colPivHouseholderQr().solve(Q_ux(time, V_xx));

        _ff_k[time] = -1 * 1 / Q_uu(time, V_xx) * (Q_u(time, V_x));
        _fb_K[time] = -1 * 1 / Q_uu(time, V_xx) * (Q_ux(time, V_xx));

        V_x = Q_x(time, V_x) + _fb_K[time].transpose() * Q_uu(time, V_xx) * _ff_k[time];
        V_x += _fb_K[time].transpose() * Q_u(time, V_x) + Q_ux(time, V_xx).transpose() * _ff_k[time];
        V_xx = Q_xx(time, V_xx) + _fb_K[time].transpose() * Q_uu(time, V_xx) * _fb_K[time];
        V_xx += _fb_K[time].transpose() * Q_ux(time, V_xx) + Q_ux(time, V_xx).transpose() * _fb_K[time];
        V_xx = 0.5 * (V_xx + V_xx.transpose());
    }
}


void ILQR::forward_pass()
{

    _x_traj_new.front() = _x_traj.front();
    for (auto time = 0; time < _simulation_time; ++time) {
        _u_traj[time] = _u_traj[time] + _ff_k[time] + _fb_K[time] * (_x_traj_new[time] - _x_traj[time]);
        std::cout << "ctrl is: " << _u_traj[time] <<std::endl;
//        clamp_control(_u_traj[time], max_bound, min_bound);
//        _u_traj[time] = std::clamp(_u_traj[time], min_bound, max_bound);
//        set_control_data(_d_cp, _u_traj[time]);
        _d_cp->qfrc_applied[1] = _u_traj[time];
        mj_step(_m, _d_cp);
        fill_state_vector(_d_cp, _x_traj_new[time + 1]);
    }

    auto new_total_cost = _cf.trajectory_running_cost(_x_traj_new, _u_traj);
    auto prev_total_cost = std::accumulate(_l.begin(), _l.end(), 0.0);

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
        backward_pass();
        forward_pass();
        _cached_control = _u_traj.front();
        std::rotate(_u_traj.begin(), _u_traj.begin() + 1, _u_traj.end());
        _u_traj.back() = 0;
        if (converged)
            break;
    }
}

double ILQR::get_control()
{
    return _cached_control;
}

