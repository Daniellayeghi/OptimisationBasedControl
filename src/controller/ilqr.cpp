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
    _simulated_state.reserve(_simulation_time);

    _V.reserve(_simulation_time);
    _V_x.reserve(_simulation_time);
    _V_xx.reserve(_simulation_time);

    std::fill(_V.begin(), _V.end(), 0);
    std::fill(_V_x.begin(), _V_x.end(), Eigen::Matrix<double, 6, 1>::Zero());
    std::fill(_V_xx.begin(), _V_xx.end(), Eigen::Matrix<double, 6, 6>::Zero());
}


void ILQR::forward_simulate(const mjData* d)
{
    copy_data(_m, d, _d_cp);
    for (auto time = 0; time < _simulation_time; ++time)
    {
        copy_data(_m, _d_cp, &sim_data);
        _simulated_state.push_back(sim_data);
    }
}


void ILQR::calculate_derivatives()
{
    mj_step(_m, _d_cp);
    _fd.f_x_f_u(_d_cp);
    _cf.derivatives(_d_cp);
}

void ILQR::backward_pass()
{

    auto result = _fd.get_full_derivatives().block<3, 6>(0, 0) * _V_x[0];
    std::cout << "Result: " << result << "\n";

//    for(auto time_step = _simulation_time - 2; time_step >= 0; --time_step )
//    {
//        calculate_derivatives();
//        auto result = _fd.get_full_derivatives().block<3, 6>(0, 0) * _V_x[time_step];
//    }
}
