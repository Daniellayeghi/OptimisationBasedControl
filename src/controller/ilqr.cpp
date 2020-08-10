#include <iostream>
#include <numeric>
#include "mujoco.h"
#include "ilqr.h"
#include "../src/controller/simulation_params.h"
#include "../utilities/basic_math.h"


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


    template<typename T, int ctrl_size >
    void set_control_data(mjData* data, const Eigen::Matrix<T, ctrl_size, 1>& ctrl)
    {
        for(auto row = 0; row < ctrl.rows(); ++row)
        {
            data->ctrl[row] = ctrl(row, 0);
        }
    }


    template<typename T, int state_size>
    void set_state_data(mjData* data, Eigen::Matrix<T, state_size, 1>& state) {
        for (unsigned int row = 0; row < state_size / 2; ++row)
        {
            data->qpos[row] = state(row, 0);
            data->qvel[row] = state(row+state_size/2, 0);
        }
    }


    template<typename T, int state_size>
    void fill_state_vector(const mjData* data, Eigen::Matrix<T, state_size, 1>& state, const mjModel* m) {
        for (unsigned int row = 0; row < state_size / 2; ++row) {
            state(row + state_size / 2, 0) = data->qvel[row];
        }

        for (unsigned int row = 0; row < state_size / 2; ++row) {
            state(row, 0) = data->qpos[row];

            int jid = m->dof_jntid[row];
            if (m->jnt_type[jid] == mjJNT_HINGE)
                state(row, 0) = BasicMath::wrap_to_2pi(data->qpos[row]);
        }
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
                                  const int simulation_time,
                                  const int iteration,
                                  const mjData* d,
                                  const std::vector<ILQR<state_size, ctrl_size>::ctrl_vec>* init_u) :
_fd(fd) ,_cf(cf), _m(m), _simulation_time(simulation_time), _iteration(iteration)
{
    _d_cp = mj_makeData(m);
    _prev_total_cost = 0;
    _regularizer.setIdentity();

    _l.assign(_simulation_time + 1, 0);
    _l_x.assign(simulation_time + 1, ilqr_t::state_vec::Zero());
    _l_xx.assign(simulation_time + 1, ilqr_t::state_mat::Zero());
    _l_u.assign(simulation_time, ilqr_t::ctrl_vec::Zero());
    _l_ux.assign(simulation_time, ilqr_t::ctrl_state_mat::Zero());
    _l_uu.assign(simulation_time, ilqr_t::ctrl_mat::Zero());

    _f_x.assign(simulation_time, ilqr_t::state_mat::Zero());
    _f_u.assign(simulation_time, ilqr_t::state_ctrl_mat::Zero());

    _fb_K.assign(_simulation_time, ilqr_t::ctrl_state_mat ::Zero());
    _ff_k.assign(_simulation_time, ilqr_t::ctrl_vec::Zero());

    _x_traj_new.assign(_simulation_time + 1, ilqr_t::state_vec::Zero());
    _x_traj.assign(_simulation_time + 1, ilqr_t::state_vec::Zero());
    _u_traj_new.assign(_simulation_time, ilqr_t::ctrl_vec::Zero());

    if(init_u == nullptr)
    {
        _u_traj.assign(_simulation_time,ilqr_t::ctrl_vec::Random() * 0);
    }else
    {
        _u_traj = *init_u;
    }

    copy_data(m, d, _d_cp);
    fill_state_vector(d, _x_traj.front(), _m);
    for (unsigned int time = 0; time < simulation_time; ++time)
    {
        set_control_data(_d_cp, _u_traj[time]);
        mj_step(m, _d_cp);
        fill_state_vector(_d_cp, _x_traj[time+1], _m);
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
        _l_xx.back() = _cf.Lf_xx(_d_cp);
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

    for (auto time = _simulation_time - 1; time >= 0; --time)
    {
        auto Qx = Q_x(time, V_x);    auto Qu  = Q_u(time, V_x);
        auto Quu = Q_uu(time, V_xx); auto Qux = Q_ux(time, V_xx); auto Qxx = Q_xx(time, V_xx);
        _ff_k[time] = -1 * Quu.colPivHouseholderQr().solve(Qu);
        _fb_K[time] = -1 * Quu.colPivHouseholderQr().solve(Qux);
        V_x   = Qx + (_fb_K[time].transpose() * Quu * (_ff_k[time]));
        V_x  += _fb_K[time].transpose() * Qu + Qux.transpose() * _ff_k[time];

        V_xx  = Qxx + _fb_K[time].transpose() * Quu * _fb_K[time];
        V_xx += _fb_K[time].transpose() * Qux + Qux.transpose() * _fb_K[time];

        V_xx  = 0.5 * (V_xx + V_xx.transpose());
    }
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::forward_pass(const mjData* d)
{
    for (const auto &backtracker : _backtrackers)
    {
        _u_traj_new.assign(_simulation_time, ilqr_t::ctrl_vec::Zero());

        copy_data(_m, d, _d_cp);
        _x_traj_new.front() = _x_traj.front();
        for (auto time = 0; time < _simulation_time; ++time)
        {
            _u_traj_new[time] = _u_traj[time] + (_ff_k[time] * backtracker) + _fb_K[time] * (_x_traj_new[time] - _x_traj[time]);
            clamp_control(_u_traj_new[time], max_bound, min_bound);
            set_control_data(_d_cp, _u_traj_new[time]);
            mj_step(_m, _d_cp);
            fill_state_vector(_d_cp, _x_traj_new[time + 1], _m);
        }
#if 1
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
            break;
        }
    }
    if (not accepted)
    {
        _delta = std::max(1.0, _delta) * _delta_init;
        static const auto min = ILQR::state_mat::Identity() * 1e-6;
        // All elements are equal hence the (0, 0) comparison
        if ((_regularizer * _delta)(0, 0) > 1e-6)
        {
            _regularizer = _regularizer * _delta;
        }
        else{
            _regularizer = min;
        }
    }
#endif
}


template<int state_size, int ctrl_size>
void ILQR<state_size, ctrl_size>::control(const mjData* d)
{
    _delta = _delta_init;
    _regularizer.setIdentity();
    recalculate = true; converged = false;
    for(auto iteration = 0; iteration < _iteration; ++iteration)
    {
        accepted = false;
        fill_state_vector(d, _x_traj.front(), _m);
        forward_simulate(d);
        backward_pass();
        forward_pass(d);
        _cached_control = _u_traj.front();
        std::rotate(_u_traj.begin(), _u_traj.begin() + 1, _u_traj.end());
        _u_traj.back() = Eigen::Matrix<double, ctrl_size, 1>::Zero();
        if (converged)
            break;
    }
    cost.emplace_back(_prev_total_cost);
    std::cout << _prev_total_cost << std::endl;
}


template class ILQR<n_jpos + n_jvel, n_ctrl>;
