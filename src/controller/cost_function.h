
#ifndef OPTCONTROL_MUJOCO_COST_FUNCTION_H
#define OPTCONTROL_MUJOCO_COST_FUNCTION_H

#include <functional>
#include <mujoco.h>
#include "eigen3/Eigen/Core"
using namespace Eigen;
#include "../utilities/internal_types.h"

using namespace InternalTypes;

template<int state_size, int ctrl_size>
class CostFunction
{
    using state_vec = Eigen::Matrix<double, state_size, 1>;
    using ctrl_vec  = Eigen::Matrix<double, ctrl_size, 1>;
    using state_mat = Eigen::Matrix<double, state_size, state_size>;
    using ctrl_mat  = Eigen::Matrix<double, ctrl_size, ctrl_size>;
    using state_ctrl_mat = Eigen::Matrix<double, ctrl_size, state_size>;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // TODO: Pass cost as functor
    CostFunction(const mjData* d,
                 const state_vec& x_desired,
                 const ctrl_vec& u_desired,
                 const state_mat& x_gain,
                 const ctrl_mat& u_gain,
                 const state_mat& x_terminal_gain);

    state_vec L_x();
    ctrl_vec  L_u();
    state_mat L_xx();
    ctrl_mat L_uu();
    state_ctrl_mat L_ux();
    state_vec Lf_x();
    state_mat Lf_xx();

    mjtNum running_cost();
    mjtNum terminal_cost();
    mjtNum trajectory_running_cost(std::vector<state_vec> & x_trajectory, std::vector<ctrl_vec> & u_trajectory);

private:
    void update_errors();
    void update_errors(state_vec &state, ctrl_vec &ctrl);

    ctrl_vec  _u;
    state_vec _x;
    ctrl_vec  _u_error;
    state_vec _x_error;
    ctrl_vec  _u_desired;
    state_vec _x_desired;
    ctrl_mat  _u_gain;
    state_mat _x_gain;
    state_mat _x_terminal_gain;

public:
    const mjData* _d;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
