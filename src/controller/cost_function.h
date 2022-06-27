
#ifndef OPTCONTROL_MUJOCO_COST_FUNCTION_H
#define OPTCONTROL_MUJOCO_COST_FUNCTION_H

#include <mujoco/mujoco.h>
#include "Eigen/Core"
#include "../parameters/simulation_params.h"
#include "../utilities/generic_utils.h"
#include <functional>

using namespace GenericUtils;


using namespace SimulationParameters;
class CostFunction
{
public:
    // TODO: Pass cost as functor
    CostFunction(const StateVector& x_desired,
                 const CtrlVector& u_desired,
                 const StateMatrix& x_gain,
                 const CtrlMatrix& u_gain,
                 const CtrlMatrix& u_diff_gain,
                 const StateMatrix& x_terminal_gain,
                 const mjModel* model);

    StateVector L_x(const mjData *d);
    CtrlVector L_u(const mjData *d);
    StateMatrix L_xx(const mjData *d);
    CtrlMatrix L_uu(const mjData *d);
    CtrlStateMatrix L_ux(const mjData *d);
    StateVector Lf_x(const mjData *d);
    StateMatrix Lf_xx();

    mjtNum running_cost(const mjData *d);
    mjtNum terminal_cost(const mjData *d);

    mjtNum trajectory_running_cost(const std::vector<StateVector>& x_trajectory,
                                   const std::vector<CtrlVector> & u_trajectory);

    void compute_value(const std::vector<CtrlVector>& u_traj,
                             const std::vector<StateVector>& x_traj,
                             std::vector<double>& state_value_vec) const;

    CtrlVector m_u_prev = CtrlVector::Zero();
private:

    void update_errors(const mjData *d);

    void update_errors(const StateVector &state,
                       const CtrlVector &ctrl);

    void compute_traj_inst_cost(std::vector<double> &inst_cost,
                                const std::vector<StateVector>& x_traj,
                                const std::vector<CtrlVector>& u_traj) const;
    // State and state errors
    CtrlVector m_u = CtrlVector::Zero();
    StateVector m_x = StateVector::Zero();
    CtrlVector m_u_error = CtrlVector::Zero();
    CtrlVector m_du_error = CtrlVector::Zero();
    StateVector m_x_error = StateVector::Zero();

    // Cost gains
    const CtrlMatrix& m_u_gain;
    const CtrlMatrix& m_u_diff_gain;
    const StateMatrix& m_x_gain;
    const StateMatrix& m_x_terminal_gain;
    const CtrlVector& m_u_desired;
    const StateVector& m_x_desired;
    const mjModel* m_m;
};

#endif //OPTCONTROL_MUJOCO_COST_FUNCTION_H
