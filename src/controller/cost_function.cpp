#include <iostream>
#include "cost_function.h"
#include "../parameters/simulation_params.h"
#include "../utilities/basic_math.h"


namespace
{

    template<int state_size, int ctrl_size>
    void fill_data(Eigen::Matrix<mjtNum , ctrl_size, 1>& u, Eigen::Matrix<mjtNum, state_size, 1>&x, const mjData* state, const mjModel* m)
    {

        for(unsigned int row = 0; row < state_size/2; ++row)
        {
            x(row + state_size/2, 0) = state->qvel[row];
        }

        for(unsigned int row = 0; row < state_size/2; ++row)
        {
            x(row, 0) = state->qpos[row];

            int jid = m->dof_jntid[row];
            if(m->jnt_type[jid] == mjJNT_HINGE)
                x(row, 0) = BasicMath::wrap_to_2pi(state->qpos[row]);
        }

        for(unsigned int row = 0; row < ctrl_size; ++row)
        {
            u(row, 0) = state->ctrl[row];
        }
    }

    std::vector<torch::jit::IValue> inputs;
}


using namespace InternalTypes;

template<int state_size, int ctrl_size>
CostFunction<state_size, ctrl_size>::CostFunction(const state_vec& x_desired,
                                                  const ctrl_vec& u_desired,
                                                  const state_mat& x_gain,
                                                  const ctrl_mat& u_gain,
                                                  const state_mat& x_terminal_gain,
                                                  const mjModel* m, torch::jit::script::Module * module) :
                                                  _m(m),
                                                  _module(module)
{
    _u_desired = u_desired;
    _x_desired = x_desired;
    _x_gain = x_gain;
    _u_gain = u_gain;
    _x_terminal_gain = x_terminal_gain;
    inputs.emplace_back(torch::empty({1, 2}));
}


template<int state_size, int ctrl_size>
void CostFunction<state_size, ctrl_size>::update_errors(const mjData *d)
{
    fill_data(_u, _x, d, _m);
    _x_error = _x - _x_desired;
    _u_error = _u;
}


template<int state_size, int ctrl_size>
inline void CostFunction<state_size, ctrl_size>::update_errors(state_vec& state,
                                                               ctrl_vec& ctrl)
{
    for(unsigned int row = 0; row < state_size/2; ++row)
    {
        int jid = _m->dof_jntid[row];
        if(_m->jnt_type[jid] == mjJNT_HINGE)
            state(row, 0) = BasicMath::wrap_to_2pi(state(row, 0));
    }
    _x_error = state - _x_desired;

    _u_error = ctrl;
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::running_cost(const mjData *d)
{
    update_errors(d);
    return (_x_error.transpose() * _x_gain * _x_error)(0,0) +
           (_u_error.transpose() * _u_gain * _u_error)(0, 0);
}


template<int state_size, int ctrl_size>
inline mjtNum CostFunction<state_size, ctrl_size>::trajectory_running_cost(std::vector<state_vec>& x_trajectory,
                                                                           std::vector<ctrl_vec>& u_trajectory)
{
    auto cost = 0.0;
    //Compute running cost
    for(unsigned int row = 0; row < u_trajectory.size(); ++row)
    {
        update_errors(x_trajectory[row], u_trajectory[row]);
        cost += (_x_error.transpose() * _x_gain * _x_error)(0,0) +
                (_u_error.transpose() * _u_gain * _u_error)(0,0);
    }

    //Compute terminal cost
    for(unsigned int row = 0; row < state_size/2; ++row)
    {
        int jid = _m->dof_jntid[row];
        if(_m->jnt_type[jid] == mjJNT_HINGE)
            x_trajectory.back()(row, 0) = BasicMath::wrap_to_2pi(x_trajectory.back()(row, 0));
    }
    _x_error =  _x_desired - x_trajectory.back();
    //Running cost + terminal cost
    return cost + (_x_error.transpose() * _x_terminal_gain * _x_error)(0, 0);
}


template<int state_size, int ctrl_size>
mjtNum CostFunction<state_size, ctrl_size>::terminal_cost(const mjData *d)
{

    update_errors(d);
    mjtNum final_cost = (_x_error.transpose() * _x_terminal_gain * _x_error)(0, 0);
    if (_module)
    {
        using namespace SimulationParameters;
        Eigen::Matrix<float, n_jpos+n_jvel, 1> x_f = _x.template cast<float>();
        inputs.front() = torch::from_blob(x_f.data(), {1, 2});
        at::Tensor value = _module->forward(inputs).toTensor();
        auto value_a = value.accessor<float, 2>();
        final_cost += final_cost + value_a[0][0];
    }

    std::cout << final_cost << "\n";
    return final_cost;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, 1> CostFunction<state_size, ctrl_size>::Lf_x(const mjData *d)
{
    update_errors(d);
    Eigen::Matrix<double, state_size, 1> v_d; v_d.setZero();

    if (_module)
    {
        auto wrap_val = [&](Eigen::Matrix<double, state_size, 1> & state_vec) {
            using namespace SimulationParameters;
            Eigen::Matrix<float, n_jpos + n_jvel, 1> x_f = state_vec.template cast<float>();
            inputs.front() = torch::from_blob(x_f.data(), {1, 2});
            at::Tensor value = _module->forward(inputs).toTensor();
            auto value_a = value.accessor<float, 2>();
            return value_a[0][0];
        };

        Eigen::Matrix<double, state_size, 1> perturb_state;

        for(int state = 0; state < state_size; ++state)
        {
            perturb_state.setZero();
            perturb_state(state, 0) += 1e-8;
            perturb_state = _x + perturb_state;
            v_d(state, 0) = (wrap_val(perturb_state) - wrap_val(_x))/1e-8;
        }
    }
    std::cout << v_d << "\n";
    return  (_x_error.transpose() * (2 *_x_terminal_gain)).transpose() + v_d;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, state_size> CostFunction<state_size, ctrl_size>::Lf_xx(const mjData* d)
{
    update_errors(d);
    auto wrap_val = [&](Eigen::Matrix<double, state_size, 1> & state_vec) {
        using namespace SimulationParameters;
        Eigen::Matrix<float, n_jpos + n_jvel, 1> x_f = state_vec.template cast<float>();
        inputs.front() = torch::from_blob(x_f.data(), {1, 2});
        at::Tensor value = _module->forward(inputs).toTensor();
        auto value_a = value.accessor<float, 2>();
        return value_a[0][0];
    };


    return  2 *_x_terminal_gain;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, 1> CostFunction<state_size, ctrl_size>::L_x(const mjData *d)
{
    update_errors(d);
    return _x_error.transpose() * (2 * _x_gain);
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, state_size, state_size> CostFunction<state_size, ctrl_size>::L_xx(const mjData *d)
{
    update_errors(d);
    return 2 * _x_gain;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, ctrl_size, 1> CostFunction<state_size, ctrl_size>::L_u(const mjData *d)
{
    update_errors(d);
    return (_u_error.transpose() * (2 * _u_gain));
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, ctrl_size, ctrl_size> CostFunction<state_size, ctrl_size>::L_uu(const mjData *d)
{
    update_errors(d);
    return 2 * _u_gain;
}


template<int state_size, int ctrl_size>
Eigen::Matrix<mjtNum, ctrl_size, state_size> CostFunction<state_size, ctrl_size>::L_ux(const mjData *d)
{
    update_errors(d);
    return Eigen::Matrix<mjtNum, ctrl_size, state_size>::Zero();
}

using namespace SimulationParameters;
template class CostFunction<n_jpos + n_jvel, n_ctrl>;