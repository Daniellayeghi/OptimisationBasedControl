#include <iostream>
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
}


ILQR::ILQR(FiniteDifference& fd, CostFunction& cf, const mjModel * m, int simulation_time) :
_fd(fd) ,_cf(cf), _m(m), _simulation_time(simulation_time)
{
    _d_cp = mj_makeData(m);

    _simulated_state.reserve(_simulation_time);

    _V.reserve(_simulation_time);
    _V_x.reserve(_simulation_time);
    _V_xx.reserve(_simulation_time);

    _F_x.reserve(_simulation_time);
    _F_u.reserve(_simulation_time);

    _L_x.reserve(_simulation_time);
    _L_u.reserve(_simulation_time);
    _L_xx.reserve(_simulation_time);
    _L_ux.reserve(_simulation_time);
    _L_uu.reserve(_simulation_time);

    std::fill(_V.begin(), _V.end(), 0);
    std::fill(_V_x.begin(), _V_x.end(), Eigen::Matrix<double, 4, 1>::Zero());
    std::fill(_V_xx.begin(), _V_xx.end(), Eigen::Matrix<double, 4, 4>::Zero());

    std::fill(_F_x.begin(), _F_x.end(), Eigen::Matrix<double, 4, 4>::Zero());
    std::fill(_F_u.begin(), _F_u.end(), Eigen::Matrix<double, 4, 2>::Zero());

    std::fill(_L_x.begin(), _L_x.end(), Eigen::Matrix<double, 4, 1>::Zero());
    std::fill(_L_u.begin(), _L_u.end(), Eigen::Matrix<double, 2, 1>::Zero());
    std::fill(_L_xx.begin(), _L_xx.end(), Eigen::Matrix<double, 4, 4>::Zero());
    std::fill(_L_ux.begin(), _L_ux.end(), Eigen::Matrix<double, 2, 4>::Zero());
    std::fill(_L_uu.begin(), _L_uu.end(), Eigen::Matrix<double, 2, 2>::Zero());
}


ILQR::~ILQR()
{
    mj_deleteData(_d_cp);
}


void ILQR::forward_simulate(const mjData* d)
{
    copy_data(_m, d, _d_cp);
    for (auto time = 0; time < _simulation_time; ++time)
    {
        _fd.f_x_f_u(_d_cp);
        _cf.derivatives(_d_cp);
        _F_x[time] = (_fd.f_x(_d_cp));
        _F_u[time] = (_fd.f_u(_d_cp));
        _L_u[time] = (_cf.L_u());
        _L_x[time] = (_cf.L_x());
        _L_xx[time] = (_cf.L_xx());
        _L_ux[time] = (_cf.L_ux());
        _L_uu[time] = (_cf.L_uu());
        mj_step(_m, _d_cp);
    }
}


Eigen::Matrix<mjtNum, 4, 1> ILQR::Q_x(int time)
{
    return _L_x[time] + _F_x[time].transpose() * _V_x[time] ;
}


Eigen::Matrix<mjtNum, 2, 1> ILQR::Q_u(int time)
{
    return _L_u[time] + _F_u[time].transpose() * _V_x[time] ;
}


Eigen::Matrix<mjtNum, 4, 4> ILQR::Q_xx(int time)
{
    return _L_xx[time] + _F_x[time].transpose() * _V_xx[time] * _F_x[time];
}


Eigen::Matrix<mjtNum, 2, 4> ILQR::Q_ux(int time)
{
    return _L_ux[time] + _F_u[time].transpose() * _V_xx[time] * _F_x[time];
}


Eigen::Matrix<mjtNum, 2, 2> ILQR::Q_uu(int time)
{
    return _L_uu[time] + _F_u[time].transpose() * _V_xx[time] * _F_u[time];
}


// TODO: make data const if you can
void ILQR::backward_pass(mjData* d)
{
    forward_simulate(d);
    std::cout << "Q_x: " <<  "\n" << Q_x(0) << "\n";
    std::cout << "Q_u: " <<  "\n" << Q_u(0) << "\n";
    std::cout << "Q_xx: " <<  "\n" << Q_xx(0) << "\n";
    std::cout << "Q_ux: " <<  "\n" << Q_ux(0) << "\n";
    std::cout << "Q_uu: " <<  "\n" << Q_uu(0) << "\n";


    std::cout << "k: " <<  "\n" << -1 * Q_uu(0).colPivHouseholderQr().solve(Q_u(0)) << "\n";
    std::cout << "K: " <<  "\n" << -1 * Q_uu(0).colPivHouseholderQr().solve(Q_ux(0)) << "\n";
    for (auto time = _simulation_time - 1; time <= 0; --time)
    {

    }
}
